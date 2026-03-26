use std::slice;

use crate::{
	kernel::{blur_pass, exposure_blur, vignette, BlurPassParams, ExposureBlurParams, VignetteParams},
	params::{get_param, Params},
};

use after_effects::log;
use premiere::{
	self as pr,
	sys::{PrGPUFilterFrameDependency, PrGPUFilterFrameDependencyType, PrGPUFilterFrameDependencyType_PrGPUDependency_InputFrame},
	Property,
};
use prgpu::{DeviceHandleInit, GPURenderProperties, Pixel, PrRect, Vec3};

#[inline]
fn frames_as_slice<'a>(frames: *const pr::sys::PPixHand, frame_count: usize) -> &'a [pr::sys::PPixHand] {
	assert!(!frames.is_null(), "frames pointer was null");
	unsafe { slice::from_raw_parts(frames, frame_count) }
}

fn gpu_bytes_per_pixels(pixel_format: pr::PixelFormat) -> i32 {
	match pixel_format {
		pr::PixelFormat::GpuBgra4444_32f => 16, // float4
		pr::PixelFormat::GpuBgra4444_16f => 8,  // half4
		_ => panic!("Unsupported pixel format"),
	}
}

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
		filter: &premiere::GpuFilterData,
		render_params: premiere::RenderParams,
		query_index: &mut i32,
	) -> Result<premiere::sys::PrGPUFilterFrameDependency, premiere::Error> {
		Err(premiere::Error::None)
	}

	fn precompute(
		&self,
		filter: &premiere::GpuFilterData,
		render_params: premiere::RenderParams,
		index: i32,
		frame: premiere::sys::PPixHand,
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
		let render_properties = match GPURenderProperties::new(filter, render_params, frames, frame_count) {
			Ok(properties) => properties,
			Err(e) => return Err(e),
		};

		unsafe {
			*out_frame = render_properties.frame;
		}

		let (incoming, outgoing) = render_properties.frames;

		// Get incoming frame data
		let (incoming_data, incoming_row_bytes) = (filter.gpu_device_suite.gpu_ppix_data(incoming)?, filter.ppix_suite.row_bytes(incoming)?);
		let incoming_pitch_px = incoming_row_bytes / bytes_per_pixel;

		// Get outgoing frame data
		let (outgoing_data, outgoing_row_bytes) = (filter.gpu_device_suite.gpu_ppix_data(outgoing)?, filter.ppix_suite.row_bytes(outgoing)?);
		let outgoing_pitch_px = outgoing_row_bytes / bytes_per_pixel;

		// Get destination frame data
		let (dest_data, dest_row_bytes) = (
			filter.gpu_device_suite.gpu_ppix_data(unsafe { *out_frame })?,
			filter.ppix_suite.row_bytes(unsafe { *out_frame })?,
		);
		let dest_pitch_px = dest_row_bytes / bytes_per_pixel;

		if incoming_data.is_null() || outgoing_data.is_null() {
			log::error!("One of the frame data pointers is null");
			return Err(pr::Error::Fail);
		}

		unsafe {
			let device = filter.gpu_info.outDeviceHandle;
			let queue = filter.gpu_info.outCommandQueueHandle;
		}

		let configuration = prgpu::Configuration {
			device_handle: device,
			context_handle: Some(filter.gpu_info.outContextHandle),
			command_queue_handle: queue,
			outgoing_data,
			incoming_data,
			dest_data,
			outgoing_pitch_px,
			incoming_pitch_px,
			dest_pitch_px,
			width: width as u32,
			height: height as u32,
			is16f,
			progress,
		};

		let user_params = VignetteParams {};

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
