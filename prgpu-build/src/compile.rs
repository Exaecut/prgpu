use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend::GpuBackend;
use crate::reflection::{self, Reflection};
use crate::sdk;

pub struct CompiledShader {
	pub metallib_path: Option<PathBuf>,
	pub msl_path: Option<PathBuf>,
	pub metal_reflection_path: Option<PathBuf>,
	pub ptx_path: Option<PathBuf>,
	pub cuda_reflection_path: Option<PathBuf>,
	pub cpp_path: PathBuf,
	pub cpu_reflection_path: PathBuf,
}

/// Compile all `.slang` shaders in `shader_dir` with vekl auto-discovered as
/// an include path. Prints rerun-if-changed hints for the shader directory
/// and every resolved include directory.
pub fn compile_shaders(
	shader_dir: &Path,
	out_dir: &Path,
	include_dirs: &[PathBuf],
	backend: GpuBackend,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
	println!("cargo:rerun-if-changed={}", shader_dir.display());
	for dir in include_dirs {
		println!("cargo:rerun-if-changed={}", dir.display());
	}

	let slang_files: Vec<PathBuf> = fs::read_dir(shader_dir)?
		.filter_map(|e| e.ok())
		.map(|e| e.path())
		.filter(|p| p.extension().and_then(|s| s.to_str()) == Some("slang"))
		.collect();

	if slang_files.is_empty() {
		return Ok(());
	}

	let sdk_path = sdk::sdk_dir();
	let slangc = sdk::slangc_bin(&sdk_path);
	if !slangc.exists() {
		panic!(
			"slangc not found at {}. Slang SDK v{} auto-download failed.",
			slangc.display(),
			sdk::SLANG_VERSION
		);
	}

	let mut cpu_cpp_paths: Vec<PathBuf> = Vec::new();

	for slang_file in &slang_files {
		let name = slang_file.file_stem().unwrap().to_str().unwrap().to_string();

		let compiled = compile_shader(&sdk_path, slang_file, &name, out_dir, include_dirs);

		validate_entry_point(&name, &compiled.cpu_reflection_path, slang_file)?;

		let user_params_size = user_params_size(&compiled.cpu_reflection_path, &name);
		write_abi_rs(out_dir, &name, user_params_size);

		copy_uniform_artifact(out_dir, &name, backend, &compiled);

		cpu_cpp_paths.push(compiled.cpp_path.clone());

		let bridge_path = crate::cpu_dispatch::generate_bridge(&name, &load_reflection(&compiled.cpu_reflection_path)?, &sdk_path, out_dir);
		cpu_cpp_paths.push(bridge_path);

		write_bindings(out_dir, &name, &compiled)?;
	}

	let cpu_paths_refs: Vec<&Path> = cpu_cpp_paths.iter().map(|p| p.as_path()).collect();
	crate::cpu_dispatch::compile_cpu_all(&cpu_paths_refs, &sdk_path);

	Ok(())
}

/// Resolve the effective include directories for Slang compilation.
/// `shader_dir` is always the first include path; vekl is probed from the
/// consumer workspace, the prgpu workspace, and the vendored copy.
pub fn resolve_include_dirs(
	shader_dir: &Path,
	extra_include: Option<&Path>,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
	let mut include_dirs = vec![shader_dir.to_path_buf()];

	if let Some(extra) = extra_include {
		include_dirs.push(extra.to_path_buf());
	}

	// 1. Consumer workspace sibling: CARGO_MANIFEST_DIR at build-script runtime
	//    is the consuming crate's manifest dir.
	if let Ok(consumer_dir) = std::env::var("CARGO_MANIFEST_DIR") {
		if let Some(parent) = PathBuf::from(&consumer_dir).parent() {
			let candidate = parent.join("vekl");
			if candidate.is_dir() {
				include_dirs.push(candidate);
				return Ok(include_dirs);
			}
		}
	}

	// 2. prgpu workspace sibling: captured at prgpu-build compile time.
	let prgpu_build_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	if let Some(parent) = prgpu_build_dir.parent().and_then(|p| p.parent()) {
		let candidate = parent.join("vekl");
		if candidate.is_dir() {
			include_dirs.push(candidate);
			return Ok(include_dirs);
		}
	}

	// 3. Vendored copy inside prgpu's own directory.
	let prgpu_dir = prgpu_build_dir.parent().unwrap_or(&prgpu_build_dir);
	let vendored = prgpu_dir.join("vekl");
	if vendored.is_dir() {
		include_dirs.push(vendored);
	}

	Ok(include_dirs)
}

fn load_reflection(path: &Path) -> Result<Reflection, Box<dyn std::error::Error + Send + Sync>> {
	let json = fs::read_to_string(path)?;
	Ok(reflection::parse_reflection(&json)?)
}

fn validate_entry_point(
	name: &str,
	cpu_reflection_path: &Path,
	slang_file: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
	let refl = load_reflection(cpu_reflection_path)?;
	let found: Vec<String> = refl.entry_points.iter().map(|ep| ep.name.clone()).collect();
	if !found.iter().any(|ep| ep == name) {
		return Err(format!(
			"{}: no compute entry point named `{name}` — the entry point must match the file name (found: {:?})",
			slang_file.display(),
			found
		)
		.into());
	}
	Ok(())
}

fn user_params_size(cpu_reflection_path: &Path, _name: &str) -> usize {
	let refl = match load_reflection(cpu_reflection_path) {
		Ok(r) => r,
		Err(_) => return usize::MAX,
	};

	let ep = match refl.entry_points.first() {
		Some(ep) => ep,
		None => return usize::MAX,
	};

	for param in &ep.parameters {
		if param.binding.as_ref().map_or(false, |b| b.kind == "constantBuffer") {
			if let Some(size) = param.binding.as_ref().and_then(|b| b.size) {
				return size as usize;
			}
		}
	}

	usize::MAX
}

fn write_abi_rs(out_dir: &Path, name: &str, user_params_size: usize) {
	let path = out_dir.join(format!("{name}.abi.rs"));
	let contents = format!("pub const USER_PARAMS_SIZE: usize = {user_params_size};\n");
	fs::write(&path, contents).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

pub fn copy_uniform_artifact(
	out_dir: &Path,
	name: &str,
	backend: GpuBackend,
	compiled: &CompiledShader,
) {
	let dest = out_dir.join(format!("{name}.shader"));
	match backend {
		GpuBackend::Metal => {
			if let Some(src) = &compiled.metallib_path {
				if let Err(e) = fs::copy(src, &dest) {
					println!("cargo:warning=[slang] {name}: failed to copy metallib to .shader: {e}");
				}
			} else {
				fs::write(&dest, []).ok();
			}
		}
		GpuBackend::Cuda => {
			if let Some(src) = &compiled.ptx_path {
				if let Err(e) = fs::copy(src, &dest) {
					println!("cargo:warning=[slang] {name}: failed to copy PTX to .shader: {e}");
				}
			} else {
				fs::write(&dest, []).ok();
			}
		}
		GpuBackend::None => {
			fs::write(&dest, []).ok();
		}
	}
}

fn write_bindings(
	out_dir: &Path,
	name: &str,
	compiled: &CompiledShader,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
	let cpu_refl = load_reflection(&compiled.cpu_reflection_path)?;
	let mut all_bindings = String::from("// Auto-generated by prgpu build from slangc -reflection-json\n\n");

	if let Some(metal_ref_path) = &compiled.metal_reflection_path {
		if let Ok(refl) = load_reflection(metal_ref_path) {
			all_bindings.push_str("// --- Metal target bindings ---\n");
			all_bindings.push_str(&crate::bindings::generate_bindings(&refl, &format!("METAL_{name}")));
			all_bindings.push('\n');
		}
	}

	if let Some(cuda_ref_path) = &compiled.cuda_reflection_path {
		if let Ok(refl) = load_reflection(cuda_ref_path) {
			all_bindings.push_str("// --- CUDA target bindings ---\n");
			all_bindings.push_str(&crate::bindings::generate_bindings(&refl, &format!("CUDA_{name}")));
			all_bindings.push('\n');
		}
	}

	all_bindings.push_str("// --- CPU target bindings ---\n");
	all_bindings.push_str(&crate::bindings::generate_bindings(&cpu_refl, "CPU"));

	let bindings_path = out_dir.join(format!("{name}_bindings.rs"));
	fs::write(&bindings_path, &all_bindings)?;

	if std::env::var_os("PRGPU_BUILD_VERBOSE").is_some() {
		println!("cargo:warning=[slang] Binding map written to: {}", bindings_path.display());
	}

	Ok(())
}

/// Slang's `-target cpp` emits vekl helpers (`FromRGBA_0`, `LoadPixel_0`, …) with
/// external linkage. Two shaders in the same crate that both pull in vekl produce
/// LNK2005 duplicates when their object files are linked into a single static lib.
fn wrap_in_anonymous_namespace(cpp_path: &Path) {
	let content = fs::read_to_string(cpp_path)
		.unwrap_or_else(|e| panic!("failed to read {} for namespace wrapping: {e}", cpp_path.display()));

	if content.contains("// prgpu: wrapped in anonymous namespace") {
		return;
	}

	let split_idx = if content.starts_with("#line ") {
		0
	} else {
		match content.find("\n#line ") {
			Some(idx) => idx + 1,
			None => {
				if std::env::var_os("PRGPU_BUILD_VERBOSE").is_some() {
					println!(
						"cargo:warning=[slang] {} has no #line directive; skipping anonymous-namespace wrap",
						cpp_path.display()
					);
				}
				return;
			}
		}
	};

	let (header, body) = content.split_at(split_idx);

	let mut wrapped = String::with_capacity(content.len() + 96);
	wrapped.push_str(header);
	wrapped.push_str("// prgpu: wrapped in anonymous namespace to dedupe vekl helpers across TUs\n");
	wrapped.push_str("namespace {\n");
	wrapped.push_str(body);
	if !body.ends_with('\n') {
		wrapped.push('\n');
	}
	wrapped.push_str("} // anonymous namespace\n");

	fs::write(cpp_path, wrapped)
		.unwrap_or_else(|e| panic!("failed to rewrite {}: {e}", cpp_path.display()));
}

fn run_slangc(sdk_path: &Path, args: &[&OsStr]) -> String {
	let slangc = sdk::slangc_bin(sdk_path);
	let output = Command::new(&slangc)
		.args(args)
		.env("SLANG_DIR", sdk_path)
		.output()
		.unwrap_or_else(|e| panic!("Failed to run slangc at {}: {e}", slangc.display()));

	if !output.status.success() {
		panic!("slangc failed:\n{}", String::from_utf8_lossy(&output.stderr));
	}

	String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Separate invocations per target for correct per-target reflection.
pub fn compile_shader(
	sdk_path: &Path,
	slang_file: &Path,
	entry_name: &str,
	out_dir: &Path,
	include_dirs: &[PathBuf],
) -> CompiledShader {
	let name = slang_file.file_stem().unwrap().to_str().unwrap().to_string();

	let include_args: Vec<&OsStr> = include_dirs
		.iter()
		.flat_map(|dir| [OsStr::new("-I"), dir.as_os_str()])
		.collect();

	let (metallib_path, msl_path, metal_reflection_path) = if cfg!(target_os = "macos") {
		let metallib = out_dir.join(format!("{name}.metallib"));
		let msl = out_dir.join(format!("{name}.metal"));
		let reflection = out_dir.join(format!("{name}_metal_reflection.json"));

		let mut args: Vec<&OsStr> = vec![
			OsStr::new("-target"), OsStr::new("metal"),
			OsStr::new("-target"), OsStr::new("metallib"),
			OsStr::new("-entry"), OsStr::new(entry_name),
			OsStr::new("-o"), msl.as_os_str(),
			OsStr::new("-o"), metallib.as_os_str(),
			OsStr::new("-reflection-json"), reflection.as_os_str(),
		];
		args.extend(&include_args);
		args.push(slang_file.as_os_str());
		run_slangc(sdk_path, &args);

		if std::env::var_os("PRGPU_BUILD_VERBOSE").is_some() {
			let ml = fs::metadata(&metallib).map(|m| m.len()).unwrap_or(0);
			let ms = fs::metadata(&msl).map(|m| m.len()).unwrap_or(0);
			println!("cargo:warning=[slang] {name}: metallib {ml} bytes, MSL {ms} bytes");
		}

		(Some(metallib), Some(msl), Some(reflection))
	} else {
		(None, None, None)
	};

	let (ptx_path, cuda_reflection_path) = if cfg!(target_os = "windows") {
		let ptx = out_dir.join(format!("{name}.ptx"));
		let reflection = out_dir.join(format!("{name}_cuda_reflection.json"));

		let mut args: Vec<&OsStr> = vec![
			OsStr::new("-target"), OsStr::new("ptx"),
			OsStr::new("-entry"), OsStr::new(entry_name),
			OsStr::new("-o"), ptx.as_os_str(),
			OsStr::new("-reflection-json"), reflection.as_os_str(),
		];
		args.extend(&include_args);
		args.push(slang_file.as_os_str());

		match Command::new(sdk::slangc_bin(sdk_path)).args(&args).env("SLANG_DIR", sdk_path).output() {
			Ok(output) if output.status.success() && ptx.exists() => {
				if std::env::var_os("PRGPU_BUILD_VERBOSE").is_some() {
					let sz = fs::metadata(&ptx).map(|m| m.len()).unwrap_or(0);
					println!("cargo:warning=[slang] {name}: PTX {sz} bytes");
				}
				(Some(ptx), Some(reflection))
			}
			_ => {
				if std::env::var_os("PRGPU_BUILD_VERBOSE").is_some() {
					println!("cargo:warning=[slang] {name}: PTX skipped (no CUDA toolkit)");
				}
				(None, None)
			}
		}
	} else {
		(None, None)
	};

	let cpp_path = out_dir.join(format!("{name}_cpu.cpp"));
	let cpu_reflection_path = out_dir.join(format!("{name}_cpu_reflection.json"));

	let mut args: Vec<&OsStr> = vec![
		OsStr::new("-target"), OsStr::new("cpp"),
		OsStr::new("-entry"), OsStr::new(entry_name),
		OsStr::new("-o"), cpp_path.as_os_str(),
		OsStr::new("-reflection-json"), cpu_reflection_path.as_os_str(),
	];
	args.extend(&include_args);
	args.push(slang_file.as_os_str());
	run_slangc(sdk_path, &args);

	wrap_in_anonymous_namespace(&cpp_path);

	CompiledShader {
		metallib_path,
		msl_path,
		metal_reflection_path,
		ptx_path,
		cuda_reflection_path,
		cpp_path,
		cpu_reflection_path,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn write_tmp(name: &str, content: &str) -> PathBuf {
		let path = std::env::temp_dir().join(format!("prgpu_wrap_test_{name}.cpp"));
		fs::write(&path, content).expect("write tmp");
		path
	}

	const PRELUDE_HEADER: &str = "#include \"slang-cpp-prelude.h\"\n\n#ifdef SLANG_PRELUDE_NAMESPACE\nusing namespace SLANG_PRELUDE_NAMESPACE;\n#endif\n\n";
	const SLANG_BODY: &str = "#line 29 \"vekl/texture/descriptor.slang\"\nstruct TextureDesc_0 { uint32_t w; };\nvoid LoadPixel_0() {}\n";

	#[test]
	fn wraps_body_after_first_line_directive() {
		let path = write_tmp("normal", &(PRELUDE_HEADER.to_string() + SLANG_BODY));
		wrap_in_anonymous_namespace(&path);
		let out = fs::read_to_string(&path).unwrap();

		assert!(out.starts_with(PRELUDE_HEADER), "prelude must remain at file scope");
		assert!(out.contains("namespace {\n#line 29"), "anonymous namespace must open immediately before #line");
		assert!(out.trim_end().ends_with("} // anonymous namespace"), "anonymous namespace must close at EOF");
		fs::remove_file(&path).ok();
	}

	#[test]
	fn handles_leading_line_directive_with_no_header() {
		let path = write_tmp("leading", SLANG_BODY);
		wrap_in_anonymous_namespace(&path);
		let out = fs::read_to_string(&path).unwrap();

		assert!(out.starts_with("// prgpu: wrapped"), "marker must lead the file when there is no header");
		assert!(out.contains("namespace {\n#line 29"));
		assert!(out.trim_end().ends_with("} // anonymous namespace"));
		fs::remove_file(&path).ok();
	}

	#[test]
	fn idempotent_when_already_wrapped() {
		let path = write_tmp("idempotent", &(PRELUDE_HEADER.to_string() + SLANG_BODY));
		wrap_in_anonymous_namespace(&path);
		let first = fs::read_to_string(&path).unwrap();
		wrap_in_anonymous_namespace(&path);
		let second = fs::read_to_string(&path).unwrap();
		assert_eq!(first, second, "second wrap must be a no-op");
		assert_eq!(first.matches("namespace {").count(), 1);
		assert_eq!(first.matches("} // anonymous namespace").count(), 1);
		fs::remove_file(&path).ok();
	}

	#[test]
	fn skips_when_no_line_directive_present() {
		let raw = "#include \"foo.h\"\nint main() { return 0; }\n";
		let path = write_tmp("no_line", raw);
		wrap_in_anonymous_namespace(&path);
		let out = fs::read_to_string(&path).unwrap();
		assert_eq!(out, raw, "files without #line must be left untouched");
		fs::remove_file(&path).ok();
	}
}
