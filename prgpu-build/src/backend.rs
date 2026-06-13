use std::env;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuBackend {
	Metal,
	Cuda,
	None,
}

impl GpuBackend {
	pub fn as_str(self) -> &'static str {
		match self {
			GpuBackend::Metal => "metal",
			GpuBackend::Cuda => "cuda",
			GpuBackend::None => "none",
		}
	}
}

/// Deterministic: TARGET + optional GPU_BACKEND env override.
/// Used by prgpu's own build.rs AND by every effect — both sides always agree
/// because both call this exact function with the same inputs.
pub fn resolve_backend() -> GpuBackend {
	let target = env::var("TARGET").expect("TARGET env var missing");
	let is_windows = target.contains("windows");
	let is_apple = target.contains("apple-darwin") || target.contains("apple-ios");

	let backend = if is_apple {
		GpuBackend::Metal
	} else if is_windows {
		GpuBackend::Cuda
	} else {
		GpuBackend::None
	};

	if let Ok(overridden) = env::var("GPU_BACKEND") {
		match overridden.to_ascii_lowercase().as_str() {
			"metal" => GpuBackend::Metal,
			"cuda" => GpuBackend::Cuda,
			"none" => GpuBackend::None,
			other => panic!("GPU_BACKEND must be 'metal', 'cuda', or 'none'; got '{other}'"),
		}
	} else {
		backend
	}
}

pub fn emit_backend_cfg(b: GpuBackend) {
	println!("cargo:rustc-check-cfg=cfg(gpu_backend, values(\"metal\", \"cuda\", \"none\"))");
	println!("cargo:rustc-cfg=gpu_backend=\"{}\"", b.as_str());
	println!("cargo:rerun-if-env-changed=GPU_BACKEND");
}
