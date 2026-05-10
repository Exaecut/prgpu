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

/// Slang's `-target cpp` emits vekl helpers (`FromRGBA_0`, `LoadPixel_0`, …) with
/// external linkage. Two shaders in the same crate that both pull in vekl produce
/// LNK2005 duplicates when their object files are linked into a single static lib.
///
/// Wrap the slang-generated body in an anonymous namespace: helpers become
/// TU-local, while `extern "C" SLANG_PRELUDE_EXPORT` entry points (`{name}`,
/// `{name}_Thread`, `{name}_Group`) keep C linkage and remain reachable from the
/// bridge .cpp via the linker.
///
/// The split point is the first `#line` directive — everything above it is the
/// prelude include + namespace using-directive that must stay at file scope.
fn wrap_in_anonymous_namespace(cpp_path: &Path) {
	let content = fs::read_to_string(cpp_path)
		.unwrap_or_else(|e| panic!("failed to read {} for namespace wrapping: {e}", cpp_path.display()));

	if content.contains("// prgpu: wrapped in anonymous namespace") {
		return;
	}

	// `#line` may sit at byte 0 (no header) or after a header that ends in `\n`.
	let split_idx = if content.starts_with("#line ") {
		0
	} else {
		match content.find("\n#line ") {
			Some(idx) => idx + 1,
			None => {
				println!(
					"cargo:warning=[slang] {} has no #line directive; skipping anonymous-namespace wrap",
					cpp_path.display()
				);
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

	wrap_in_anonymous_namespace(&cpp_path);

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

		// Header (prelude + using directive) stays at file scope.
		assert!(out.starts_with(PRELUDE_HEADER), "prelude must remain at file scope");
		// Marker + namespace open sit between header and body.
		assert!(out.contains("namespace {\n#line 29"), "anonymous namespace must open immediately before #line");
		// Closer at the end.
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
		// Exactly one anonymous namespace pair.
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
