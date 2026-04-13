pub mod buffer;
pub mod render;

// Most of the CPU pipeline is handled by Adobe CPU Render Path.
// We only need to modify the pipeline to support shader hot-reload and other debugging features.
//
// Gate on `feature = "shader_hotreload"` (not bare `cfg(shader_hotreload)`) because the
// Cargo feature is what activates the `libloading` dep that pipeline.rs requires.
// `cfg(shader_hotreload)` can be emitted by prgpu/build.rs from EX_SHADER_HOTRELOAD=true
// (without the feature) to keep GPU hotreload working; in that case the stub is used for CPU.
#[cfg(shader_hotreload)]
pub mod pipeline;

// No-op stub when the full hot-reload pipeline is not compiled (either because
// shader_hotreload feature is not enabled, or EX_SHADER_HOTRELOAD without the feature).
// Allows callers such as `vignette/src/lib.rs` to call `prgpu::cpu::pipeline::hot_reload()`
// unconditionally without a cfg gate at the call site.
#[cfg(not(shader_hotreload))]
pub mod pipeline {
	use crate::cpu::render::CpuDispatchFn;

	/// No-op: CPU shader hot-reload is not enabled in this build.
	///
	/// Enable it by building with `--features shader_hotreload` (debug only).
	pub fn hot_reload() {
		after_effects::log::info!("[CPU] Hot reload not available (build does not include shader_hotreload feature).");
	}

	/// No-op: shader directory registration is a hot-reload concern only.
	#[allow(unused_variables)]
	pub fn set_shader_dirs(_shader_dir: std::path::PathBuf, _include_dirs: Vec<std::path::PathBuf>) {}

	/// No-op: always returns the statically-linked fallback unchanged.
	///
	/// In shader_hotreload builds this would resolve the runtime-compiled symbol;
	/// here the static link IS the only kernel, so there is nothing to look up.
	pub fn get_dispatch_fn(_kernel_name: &'static str, static_fallback: CpuDispatchFn) -> CpuDispatchFn {
		static_fallback
	}
}

// Shared codegen utilities: parse_kernel_signature + generate_cpu_dispatch_wrapper.
// Needed at build time (feature = "build") and at runtime for hot-reload.
#[cfg(any(feature = "build", feature = "shader_hotreload"))]
pub(crate) mod codegen;
