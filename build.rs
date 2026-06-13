use std::path::PathBuf;

fn main() {
	let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
	let shader_dir = manifest_dir.join("shaders");
	prgpu_build::compile_builtin_shaders(&shader_dir)
		.expect("built-in shader compilation failed");
}
