use std::env;

fn main() {
	let target = env::var("TARGET").expect("TARGET env var missing");

	let is_windows = target.contains("windows");
	let is_apple = target.contains("apple-darwin") || target.contains("apple-ios");

	let backend = if is_apple {
		"metal"
	} else if is_windows {
		if env::var_os("CARGO_FEATURE_OPENCL").is_some() { "opencl" } else { "cuda" }
	} else {
		"other"
	};

	let backend = if let Ok(overridden) = env::var("GPU_BACKEND") {
		Box::leak(overridden.into_boxed_str())
	} else {
		backend
	};

	println!("cargo:rustc-check-cfg=cfg(gpu_backend, values(\"metal\", \"cuda\", \"opencl\", \"other\"))");

	println!("cargo:rustc-cfg=gpu_backend=\"{}\"", backend);

	let hotreload_env = env::var("EX_SHADER_HOTRELOAD").unwrap_or("false".to_string()) == "true";
	let hotreload_feature = env::var_os("CARGO_FEATURE_SHADER_HOTRELOAD").is_some();
	let shader_hotreload = hotreload_env || hotreload_feature;

	if shader_hotreload {
        println!("cargo:warning=Hot reloading shaders is enabled. This is not recommended for production builds.");
		println!("cargo:rustc-cfg=shader_hotreload");
	}

	println!("cargo:rerun-if-env-changed=GPU_BACKEND");
	println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENCL");
    println!("cargo:rerun-if-env-changed=EX_SHADER_HOTRELOAD");
	println!("cargo:rerun-if-changed=build.rs");
}
