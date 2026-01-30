use std::env;

fn main() {
    let target = env::var("TARGET").expect("TARGET env var missing");

    // Logic to determine backend
    let is_windows = target.contains("windows");
    let is_apple = target.contains("apple-darwin") || target.contains("apple-ios");

    let mut backend = if is_apple {
        "metal"
    } else if is_windows {
        // Default to cuda on windows unless opencl feature is on
        if env::var_os("CARGO_FEATURE_OPENCL").is_some() {
            "opencl"
        } else {
            "cuda"
        }
    } else {
        "other"
    };

    // Allow manual override via environment variable
    let backend = if let Ok(overridden) = env::var("GPU_BACKEND") {
        Box::leak(overridden.into_boxed_str())
    } else {
        backend
    };

    // 1. Tell the compiler this is a valid CFG molecule
    println!(
        "cargo:rustc-check-cfg=cfg(gpu_backend, values(\"metal\", \"cuda\", \"opencl\", \"other\"))"
    );

    // 2. Set the actual value for this compilation
    println!("cargo:rustc-cfg=gpu_backend=\"{}\"", backend);

    // Rebuild triggers
    println!("cargo:rerun-if-env-changed=GPU_BACKEND");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENCL");
    println!("cargo:rerun-if-changed=build.rs");
}
