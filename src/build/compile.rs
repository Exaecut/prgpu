use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::sdk;

pub struct CompiledShader {
	pub metallib_path: Option<PathBuf>,
	pub msl_path: Option<PathBuf>,
	pub metal_reflection_path: Option<PathBuf>,
	pub ptx_path: Option<PathBuf>,
	pub cuda_reflection_path: Option<PathBuf>,
	pub cpp_path: PathBuf,
	pub cpu_reflection_path: PathBuf,
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

	// Metal (macOS)
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

		let ml = fs::metadata(&metallib).map(|m| m.len()).unwrap_or(0);
		let ms = fs::metadata(&msl).map(|m| m.len()).unwrap_or(0);
		println!("cargo:warning=[slang] {name}: metallib {ml} bytes, MSL {ms} bytes");

		(Some(metallib), Some(msl), Some(reflection))
	} else {
		(None, None, None)
	};

	// CUDA PTX (Windows)
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
				let sz = fs::metadata(&ptx).map(|m| m.len()).unwrap_or(0);
				println!("cargo:warning=[slang] {name}: PTX {sz} bytes");
				(Some(ptx), Some(reflection))
			}
			_ => {
				println!("cargo:warning=[slang] {name}: PTX skipped (no CUDA toolkit)");
				(None, None)
			}
		}
	} else {
		(None, None)
	};

	// CPU (always)
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

	let sz = fs::metadata(&cpp_path).map(|m| m.len()).unwrap_or(0);
	println!("cargo:warning=[slang] {name}: C++ {sz} bytes");

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
