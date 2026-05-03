use std::path::Path;

/// Compile the Slang-generated C++ source to a static library via cc-rs.
pub fn compile_cpu_cpp(cpp_path: &Path, sdk_path: &Path) {
	let sdk_include = sdk_path.join("include");

	let mut build = cc::Build::new();
	build
		.cpp(true)
		.opt_level(2)
		.flag_if_supported("-std=c++17")
		.include(&sdk_include)
		.file(cpp_path);

	let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or("unknown".into());
	let lib_name = format!("{}_slang_cpu", pkg_name);
	build.compile(&lib_name);

	println!("cargo:warning=[slang] C++ compiled to static .a — zero runtime deps");
}
