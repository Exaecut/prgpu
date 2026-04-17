pub mod buffer;
pub mod render;

#[cfg(shader_hotreload)]
pub mod pipeline;

#[cfg(not(shader_hotreload))]
pub mod pipeline {
	use crate::cpu::render::CpuDispatchFn;

	pub fn hot_reload() {
		after_effects::log::info!("[CPU] Hot reload not available (build does not include shader_hotreload feature).");
	}

	#[allow(unused_variables)]
	pub fn set_shader_dirs(_shader_dir: std::path::PathBuf, _include_dirs: Vec<std::path::PathBuf>) {}

	pub fn get_dispatch_fn(_kernel_name: &'static str, static_fallback: CpuDispatchFn) -> CpuDispatchFn {
		static_fallback
	}
}

#[cfg(any(feature = "build", shader_hotreload))]
pub(crate) mod codegen;
