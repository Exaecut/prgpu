use std::{error::Error, path::PathBuf};

#[cfg(target_os = "windows")]
use cudarc::nvrtc::{CompileError, CompileOptions};

// Shared codegen utilities live in cpu/codegen.rs so they can be used both at
// build time (here, via `feature = "build"`) and at runtime by cpu/pipeline.rs
// (when `feature = "shader_hotreload"` is active).
use crate::cpu::codegen::{generate_cpu_dispatch_wrapper, parse_kernel_signature};
use crate::gpu::shaders::expand_includes_runtime;

type DynError = Box<dyn Error + Send + Sync>;

#[cfg(target_os = "windows")]
pub fn parse_nvrtc_error(err: &CompileError) -> String {
	match err {
		CompileError::CompileError { log, .. } => {
			let log = log.to_string_lossy();

			let mut out = String::new();
			let mut current_block = Vec::new();

			for line in log.lines() {
				let line = line.trim_end();

				if line.contains("note #") {
					continue;
				}

				if line.contains("): error:") && !current_block.is_empty() {
					out.push_str(&format_block(&current_block));
					current_block.clear();
				}

				current_block.push(line.to_string());
			}

			if !current_block.is_empty() {
				out.push_str(&format_block(&current_block));
			}

			if out.is_empty() { log.into_owned() } else { out }
		}

		other => format!("{:#?}", other),
	}
}

fn format_block(block: &[String]) -> String {
	let mut out = String::new();

	if let Some(header) = block.first()
		&& let Some((path_part, rest)) = header.split_once("): ")
		&& let Some((path, line)) = path_part.rsplit_once('(')
	{
		let file = path.split('\\').next_back().unwrap_or(path);

		out.push_str(&format!("\nerror: {}\n", rest));
		out.push_str(&format!(" --> {}:{}\n", file, line));
		out.push_str("  |\n");

		for l in &block[1..] {
			out.push_str(&format!("  {}\n", l));
		}

		return out;
	}

	for l in block {
		out.push_str(l);
		out.push('\n');
	}

	out
}

pub fn compile_shaders(shader_dir: &str) -> Result<(), DynError> {
	let out_dir = std::env::var("OUT_DIR").unwrap();

	let utils = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("vekl").canonicalize().unwrap();
	let utils_str = utils.to_string_lossy().replace("\\\\?\\", "");
	let shader_dir_abs = PathBuf::from(shader_dir).canonicalize().unwrap();
	let include_dirs = vec![utils.clone()];

	#[cfg(target_os = "windows")]
	let cuda_include = {
		let cuda_path = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")).unwrap_or("/usr/local/cuda".into());
		PathBuf::from(&cuda_path).join("include").canonicalize().unwrap()
	};

	let mut cpu_sources: Vec<(String, PathBuf)> = Vec::new();

	for entry in std::fs::read_dir(shader_dir).unwrap() {
		let path = entry.unwrap().path();

		if path.extension().and_then(|s| s.to_str()) != Some("vekl") {
			continue;
		}

		let name = path.file_stem().unwrap().to_str().unwrap().to_string();
		let src = std::fs::read_to_string(&path).unwrap();
		let expanded_metal = expand_includes_runtime(&src, &shader_dir_abs, &include_dirs).map_err(|e| format!("Failed to flatten Metal source for {}: {e}", path.display()))?;
		let metal_path = PathBuf::from(&out_dir).join(format!("{name}.metal"));
		std::fs::write(&metal_path, expanded_metal)?;
		println!("cargo:warning=Metal shader source generated -> {}", metal_path.to_str().unwrap());

		// GPU: compile f32 and f16 PTX variants via NVRTC.
		for (suffix, half_precision) in [("", false), ("_f16", true)] {
			let tag = format!("{name}{suffix}");
			let ptx_path = PathBuf::from(&out_dir).join(format!("{tag}.ptx"));

			#[cfg(target_os = "windows")]
			{
				use crate::gpu::shaders::prepare_cuda_source;
				let prepared_src = prepare_cuda_source(&src, &name);
				let mut extra_opts = vec![
					"--std=c++14".into(),
					"--extra-device-vectorization".into(),
					"--device-as-default-execution-space".into(),
					"-DVEKL_CUDA=1".into(),
				];
				if cfg!(debug_assertions) {
					extra_opts.push("-DDEBUG=1".into());
				}

				if half_precision {
					extra_opts.push("-DUSE_HALF_PRECISION=1".into());
				}

				let opts = CompileOptions {
					ftz: Some(true),
					prec_sqrt: Some(false),
					prec_div: Some(false),
					fmad: Some(true),
					use_fast_math: None,
					include_paths: vec![utils_str.clone(), cuda_include.to_string_lossy().replace("\\\\?\\", "")],
					arch: Some("compute_86"),
					options: extra_opts,
					..Default::default()
				};

				let ptx = cudarc::nvrtc::compile_ptx_with_opts(&prepared_src, opts).map_err(|e| {
					let pretty = parse_nvrtc_error(&e);
					eprintln!("Compile failed [{tag}]:\n{pretty}");
					println!("cargo:warning=Compile failed [{tag}]:\n{pretty}");
					Box::new(e) as DynError
				})?;

				let ptx_bytes = ptx.as_bytes().unwrap();
				let ptx_bytes = if ptx_bytes.last() == Some(&0) {
					&ptx_bytes[..ptx_bytes.len() - 1]
				} else {
					ptx_bytes
				};
				std::fs::write(&ptx_path, ptx_bytes)?;
				println!("cargo:warning=Shader compiled successfully to -> {}", ptx_path.to_str().unwrap());
			}

			#[cfg(not(target_os = "windows"))]
			{
				let _ = half_precision;
				std::fs::write(&ptx_path, b"")?;
				println!("cargo:warning=CUDA PTX placeholder generated -> {}", ptx_path.to_str().unwrap());
			}
		}

		// CPU: Generate dispatch wrapper .cpp
		let sig = parse_kernel_signature(&src).ok_or_else(|| format!("Failed to parse kernel signature in {}", path.display()))?;

		let shader_abs = path.canonicalize().unwrap();
		let shader_abs_str = shader_abs.to_string_lossy().replace("\\\\?\\", "").replace("//?/", "").replace("\\", "/");
		let wrapper_code = generate_cpu_dispatch_wrapper(&shader_abs_str, &sig);

		let wrapper_path = PathBuf::from(&out_dir).join(format!("{}_cpu_dispatch.cpp", name));
		std::fs::write(&wrapper_path, &wrapper_code)?;

		println!("cargo:warning=CPU dispatch wrapper generated -> {}", wrapper_path.to_str().unwrap());

		cpu_sources.push((name, wrapper_path));
	}

	// CPU: Compile all wrappers into a static library via cc.
	if !cpu_sources.is_empty() {
		let mut build = cc::Build::new();
		build
			.cpp(true)
			.opt_level(3)
			.include(&utils_str)
			.include(shader_dir_abs.to_str().unwrap())
			.define("VEKL_CPU", Some("1"))
			.flag_if_supported("/std:c++14") // MSVC
			.flag_if_supported("-std=c++14") // Clang/GCC
			.flag_if_supported("/fp:fast") // MSVC fast math
			.flag_if_supported("-ffast-math") // Clang/GCC fast math
			.flag_if_supported("/Oi") // MSVC intrinsics
			.flag_if_supported("/arch:AVX2") // MSVC SIMD
			.flag_if_supported("-mavx2") // Clang/GCC SIMD
			.flag_if_supported("-mfma"); // Clang/GCC FMA

		if cfg!(debug_assertions) {
			build.define("DEBUG", Some("1"));
		}

		for (name, wrapper_path) in &cpu_sources {
			build.file(wrapper_path);
			println!("cargo:warning=Compiling CPU kernel: {}", name);
		}

		let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or("unknown".into());
		let lib_name = format!("{}_cpu_kernels", pkg_name);
		build.compile(&lib_name);
		println!("cargo:warning=CPU shader library compiled: {}", lib_name);
	}

	Ok(())
}
