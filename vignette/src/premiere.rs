use crate::kernel::{vignette, VignetteParams};

use after_effects::log;
use premiere::{self as pr};
use prgpu::GPURenderProperties;

#[derive(Default)]
struct PremiereGPU;

impl pr::GpuFilter for PremiereGPU {
	fn global_init() {}

	fn global_destroy() {
		unsafe {
			prgpu::gpu::pipeline::cleanup();
		}
	}

	fn get_frame_dependencies(
		&self,
		_filter: &premiere::GpuFilterData,
		_render_params: premiere::RenderParams,
		_query_index: &mut i32,
	) -> Result<premiere::sys::PrGPUFilterFrameDependency, premiere::Error> {
		Err(premiere::Error::None)
	}

	fn precompute(
		&self,
		_filter: &premiere::GpuFilterData,
		_render_params: premiere::RenderParams,
		_index: i32,
		_frame: premiere::sys::PPixHand,
	) -> Result<(), premiere::Error> {
		Ok(())
	}

	fn render(
		&self,
		filter: &premiere::GpuFilterData,
		render_params: premiere::RenderParams,
		frames: *const premiere::sys::PPixHand,
		frame_count: usize,
		out_frame: *mut premiere::sys::PPixHand,
	) -> Result<(), premiere::Error> {
		let render_properties = unsafe { GPURenderProperties::new(filter, render_params, frames, frame_count, out_frame) }?;
		let configuration = unsafe { prgpu::Configuration::effect(&render_properties, out_frame)? };

		// Use get_params here.
		let user_params = VignetteParams { softness: 0.0, strength: 1.0 };

		unsafe {
			vignette(&configuration, user_params).map_err(|e| {
				log::error!("Kernel execution failed: {e}");
				pr::Error::Fail
			})?;
		}

		Ok(())
	}
}

pr::define_gpu_filter!(PremiereGPU);
