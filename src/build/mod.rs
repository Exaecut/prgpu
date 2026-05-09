use std::{error::Error, path::{Path, PathBuf}};

pub mod sdk;
pub mod compile;
pub mod reflection;
pub mod bindings;
pub mod cpu_dispatch;
pub mod lsp;

pub type DynError = Box<dyn Error + Send + Sync>;

pub use lsp::{vekl_include_path, write_slang_lsp_config};

/// Compile all `.slang` shaders in `shader_dir` with vekl auto-discovered as
/// an include path. Lookups are probed in this order — first hit wins:
///
/// 1. **Consumer workspace sibling** — `../vekl` relative to the crate that
///    called `compile_shaders` (resolved at runtime from the build script's
///    `CARGO_MANIFEST_DIR`). This is the path that lets a crate consuming
///    `prgpu` from crates.io still pick up a locally-checked-out vekl sitting
///    next to it in the user's workspace, so hot-fixes to vekl don't require
///    a republish of prgpu.
/// 2. **prgpu workspace sibling** — `../vekl` relative to prgpu's own manifest
///    (resolved at compile time via `env!`). Matches the in-tree dev layout
///    where `/prgpu/` and `/vekl/` live side-by-side.
/// 3. **Vendored copy** — `CARGO_MANIFEST_DIR/vekl` baked into prgpu's
///    published tarball. Used when prgpu is consumed from crates.io without
///    a local vekl checkout anywhere in sight.
///
/// If none of the above resolve, the shader directory is the only include
/// path — no error is raised. That lets users who ship their own shader
/// library skip vekl entirely via [`compile_shaders_with`].
///
/// Slang is always required. vekl is optional.
pub fn compile_shaders(shader_dir: &str) -> Result<(), DynError> {
	let mut include_dirs = Vec::new();

	// 1. Consumer workspace sibling: `CARGO_MANIFEST_DIR` at build-script
	//    runtime is the *consuming* crate's manifest dir (e.g. `effects/vignette/`),
	//    so its parent is the workspace root. This is the lookup that matters
	//    for local vekl dev while prgpu itself is still pinned to a release.
	if let Ok(consumer_dir) = std::env::var("CARGO_MANIFEST_DIR") {
		if let Some(parent) = PathBuf::from(&consumer_dir).parent() {
			let candidate = parent.join("vekl");
			if candidate.is_dir() {
				include_dirs.push(candidate);
			}
		}
	}

	// 2. prgpu workspace sibling: `env!` captures prgpu's own manifest dir at
	//    the time prgpu was compiled. Useful when prgpu itself builds inside
	//    the dev workspace (e.g. `cargo build -p prgpu` for its own shaders).
	if include_dirs.is_empty() {
		let prgpu_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		if let Some(parent) = prgpu_dir.parent() {
			let candidate = parent.join("vekl");
			if candidate.is_dir() {
				include_dirs.push(candidate);
			}
		}
	}

	// 3. Vendored copy inside prgpu's own directory (always present in the
	//    published tarball, see `include = ["vekl/**/*.slang", ...]` in
	//    prgpu/Cargo.toml).
	if include_dirs.is_empty() {
		let prgpu_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		let vendored = prgpu_dir.join("vekl");
		if vendored.is_dir() {
			include_dirs.push(vendored);
		}
	}

	compile_shaders_with(shader_dir, &include_dirs)
}

/// Compile all `.slang` shaders in `shader_dir` with explicit include
/// directories. Use this when you need custom include paths or want to
/// omit vekl entirely.
///
/// The shader directory is always added as the first include path.
pub fn compile_shaders_with(shader_dir: &str, include_dirs: &[PathBuf]) -> Result<(), DynError> {
	let out_dir = std::env::var("OUT_DIR").unwrap();
	let out_path = PathBuf::from(&out_dir);

	let shader_dir_abs = PathBuf::from(shader_dir).canonicalize().unwrap();

	let mut all_include = vec![shader_dir_abs.clone()];
	all_include.extend(include_dirs.iter().cloned());

	compile_slang_shaders(&shader_dir_abs, &out_path, &all_include)?;

	Ok(())
}

fn compile_slang_shaders(shader_dir: &PathBuf, out_dir: &PathBuf, include_dirs: &[PathBuf]) -> Result<(), DynError> {
	let slang_files: Vec<PathBuf> = std::fs::read_dir(shader_dir)
		.unwrap()
		.filter_map(|e| e.ok())
		.map(|e| e.path())
		.filter(|p| p.extension().and_then(|s| s.to_str()) == Some("slang"))
		.collect();

	if slang_files.is_empty() {
		return Ok(());
	}

	let sdk = sdk::sdk_dir();
	let slangc = sdk::slangc_bin(&sdk);
	if !slangc.exists() {
		panic!(
			"slangc not found at {}. Slang SDK v{} auto-download failed.\n\
			 Manually download from: https://github.com/shader-slang/slang/releases/tag/{}",
			slangc.display(), sdk::SLANG_VERSION, sdk::SLANG_TAG
		);
	}

	let mut cpu_cpp_paths: Vec<PathBuf> = Vec::new();

	for slang_file in &slang_files {
		let name = slang_file.file_stem().unwrap().to_str().unwrap().to_string();

		let compiled = compile::compile_shader(
			&sdk, slang_file, &name, out_dir, include_dirs,
		);

		cpu_cpp_paths.push(compiled.cpp_path.clone());

		let cpu_json = std::fs::read_to_string(&compiled.cpu_reflection_path)?;
		let cpu_refl = reflection::parse_reflection(&cpu_json)
			.unwrap_or_else(|e| panic!("Failed to parse CPU reflection JSON: {e}"));

		// Generate bridge wrapper
		let bridge_path = cpu_dispatch::generate_bridge(&name, &cpu_refl, &sdk, out_dir);
		cpu_cpp_paths.push(bridge_path);

		let mut all_bindings = String::from("// Auto-generated by prgpu build from slangc -reflection-json\n\n");

		if let Some(metal_ref_path) = &compiled.metal_reflection_path {
			let json = std::fs::read_to_string(metal_ref_path)?;
			let refl = reflection::parse_reflection(&json)
				.unwrap_or_else(|e| panic!("Failed to parse Metal reflection JSON: {e}"));
			all_bindings.push_str("// --- Metal target bindings ---\n");
			all_bindings.push_str(&bindings::generate_bindings(&refl, &format!("METAL_{name}")));
			all_bindings.push('\n');
		}

		if let Some(cuda_ref_path) = &compiled.cuda_reflection_path {
			let json = std::fs::read_to_string(cuda_ref_path)?;
			let refl = reflection::parse_reflection(&json)
				.unwrap_or_else(|e| panic!("Failed to parse CUDA reflection JSON: {e}"));
			all_bindings.push_str("// --- CUDA target bindings ---\n");
			all_bindings.push_str(&bindings::generate_bindings(&refl, &format!("CUDA_{name}")));
			all_bindings.push('\n');
		}

		all_bindings.push_str("// --- CPU target bindings ---\n");
		all_bindings.push_str(&bindings::generate_bindings(&cpu_refl, "CPU"));

		let bindings_path = out_dir.join(format!("{name}_bindings.rs"));
		std::fs::write(&bindings_path, &all_bindings)?;
		println!("cargo:warning=[slang] Binding map written to: {}", bindings_path.display());
	}

	// Compile all C++ sources (Slang-generated + bridge wrappers) in one pass
	let cpu_paths_refs: Vec<&Path> = cpu_cpp_paths.iter().map(|p| p.as_path()).collect();
	cpu_dispatch::compile_cpu_all(&cpu_paths_refs, &sdk);

	Ok(())
}
