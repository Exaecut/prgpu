//! Premiere GPU adapter: drives `pr::GpuFilter` through the [`Effect`] trait.
//!
//! Effects declare:
//!
//! ```ignore
//! pub type PremiereGPU = prgpu::adobe::premiere::GpuFilterAdapter<MyEffect>;
//! pr::define_gpu_filter!(PremiereGPU);
//! ```
//!
//! The adapter normalises the `GpuFilterData` + `RenderParams` + `PPixHand`
//! frames into an `InvocationBase`, builds `FrameData` via
//! `Effect::frame_data`, applies the source-snapshot policy through the
//! graph executor, then runs the cached `RenderGraph`.

use std::sync::OnceLock;

use after_effects as ae;
use after_effects::log;
use premiere::{self as pr};

use crate::effect::frame_context::{FrameDataContext, HostBackend};
use crate::effect::host::{Host, RenderKind};
use crate::effect::{Effect, FrameBinding, InvocationBase, LicenseGate, PixelLayout};
use crate::graph::{execute::execute as run_graph, RenderGraph};
use crate::gpu::pipeline;
use crate::gpu::render_properties::GPURenderProperties;
use crate::types::{Backend, Configuration};
use crate::PrRect;

/// Adobe high-precision time uses 254 016 000 000 ticks/sec; the SDK does
/// not expose this as a constant.
const PR_TICKS_PER_SECOND: f64 = 254_016_000_000.0;

pub struct GpuFilterAdapter<E: Effect> {
	license: E::License,
	graph: OnceLock<RenderGraph<E::FrameData>>,
}

impl<E: Effect> Default for GpuFilterAdapter<E> {
	fn default() -> Self {
		Self {
			license: E::License::default(),
			graph: OnceLock::new(),
		}
	}
}

impl<E: Effect> GpuFilterAdapter<E> {
	fn graph(&self) -> &RenderGraph<E::FrameData> {
		self.graph.get_or_init(|| {
			let mut g = RenderGraph::new();
			E::pipeline(&mut g);
			g
		})
	}

	fn build_invocation(props: &GPURenderProperties<'_>, base_cfg: &Configuration, w: u32, h: u32, bpp: u32) -> Result<InvocationBase, pr::Error> {
		let pixel_layout = PixelLayout::from_u32(base_cfg.pixel_layout);

		let main = FrameBinding {
			data: base_cfg.outgoing_data.unwrap_or(std::ptr::null_mut()),
			pitch_px: base_cfg.outgoing_pitch_px,
			width: w,
			height: h,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};
		let output = FrameBinding {
			data: base_cfg.dest_data,
			pitch_px: base_cfg.dest_pitch_px,
			width: w,
			height: h,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};

		let backend = match props.gpu_index {
			_ => {
				#[cfg(gpu_backend = "metal")]
				{
					Backend::Metal
				}
				#[cfg(gpu_backend = "cuda")]
				{
					Backend::Cuda
				}
				#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
				{
					Backend::Cpu
				}
			}
		};

		// CUDA needs the context handle as the device handle for buffer alloc;
		// CUDA: device handle = CUcontext (contextPV). Metal: device handle = MTLDevice (devicePV).
		#[cfg(gpu_backend = "cuda")]
		let device_handle = base_cfg.context_handle.unwrap_or(std::ptr::null_mut());
		#[cfg(gpu_backend = "metal")]
		let device_handle = base_cfg.device_handle;
		#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
		let device_handle: *mut c_void = std::ptr::null_mut();

		Ok(InvocationBase {
			host: Host::Premiere,
			backend,
			render_kind: RenderKind::PremiereGpuEffect,
			device_handle,
			context_handle: base_cfg.context_handle,
			command_queue_handle: base_cfg.command_queue_handle,
			bytes_per_pixel: bpp,
			pixel_layout,
			time: base_cfg.time,
			progress: base_cfg.progress,
			render_generation: base_cfg.render_generation,
			main_source: main,
			incoming_source: None,
			outgoing_source: None,
			output,
		})
	}
}

impl<E: Effect> pr::GpuFilter for GpuFilterAdapter<E> {
	fn global_init() {}

	fn global_destroy() {
		unsafe {
			pipeline::cleanup();
			crate::gpu::buffer::cleanup();
		}
	}

	fn get_frame_dependencies(&self, _filter: &pr::GpuFilterData, _render_params: pr::RenderParams, _query_index: &mut i32) -> Result<pr::sys::PrGPUFilterFrameDependency, pr::Error> {
		Err(pr::Error::None)
	}

	fn precompute(&self, _filter: &pr::GpuFilterData, _render_params: pr::RenderParams, _index: i32, _frame: pr::sys::PPixHand) -> Result<(), pr::Error> {
		Ok(())
	}

	fn render(&self, filter: &pr::GpuFilterData, render_params: pr::RenderParams, frames: *const pr::sys::PPixHand, frame_count: usize, out_frame: *mut pr::sys::PPixHand) -> Result<(), pr::Error> {
		if !self.license.is_valid() {
			return Ok(());
		}

		let props = unsafe { GPURenderProperties::new(filter, render_params.clone(), frames, frame_count, out_frame) }?;
		let mut base_cfg = unsafe { Configuration::effect(&props, out_frame)? };

		// Some Premiere builds give us PPix bounds that disagree with
		// `render_params.render_*()` (e.g. Premiere 25.2 native-res ppix).
		// Premiere 25.2 native-res PPix workaround: when src/dst bounds match
		// each other but differ from `bounds`, prefer them.
		let src_ppix = props.frames.1;
		let dst_ppix = unsafe { *out_frame };
		if let (Ok(sb), Ok(db)) = (filter.ppix_suite.bounds(src_ppix), filter.ppix_suite.bounds(dst_ppix)) {
			let sr = ae::Rect::from(PrRect::from(sb));
			let dr = ae::Rect::from(PrRect::from(db));
			let sw = sr.width().max(0) as u32;
			let sh = sr.height().max(0) as u32;
			let dw = dr.width().max(0) as u32;
			let dh = dr.height().max(0) as u32;
			if sw > 0 && sh > 0 && sw == dw && sh == dh && (sw, sh) != (base_cfg.width, base_cfg.height) {
				base_cfg.width = sw;
				base_cfg.height = sh;
				base_cfg.outgoing_width = sw;
				base_cfg.outgoing_height = sh;
				base_cfg.incoming_width = sw;
				base_cfg.incoming_height = sh;
			}
		}

		let w = base_cfg.width;
		let h = base_cfg.height;
		if w == 0 || h == 0 {
			return Ok(());
		}
		let bpp = props.bytes_per_pixel as u32;

		// Drop frames where Premiere reports a source pitch shorter than
		// the destination width — same defensive bail as the legacy path.
		let expected_pitch_bytes = w.saturating_mul(bpp);
		let src_pitch_bytes = (base_cfg.outgoing_pitch_px as u32).saturating_mul(bpp);
		if src_pitch_bytes < expected_pitch_bytes {
			log::warn!("[adapter] skipping frame: source pitch {src_pitch_bytes} < expected {expected_pitch_bytes}. dims={w}x{h} bpp={bpp}");
			return Ok(());
		}

		let frame_index = if render_params.render_ticks_per_frame() != 0 {
			(render_params.clip_time() / render_params.render_ticks_per_frame()) as u32
		} else {
			0
		};
		let time_seconds = (render_params.clip_time() as f64 / PR_TICKS_PER_SECOND) as f32;

		let ctx = FrameDataContext {
			host: Host::Premiere,
			backend: {
				#[cfg(gpu_backend = "cuda")]
				{
					Backend::Cuda
				}
				#[cfg(gpu_backend = "metal")]
				{
					Backend::Metal
				}
				#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
				{
					Backend::Cpu
				}
			},
			render_kind: RenderKind::PremiereGpuEffect,
			inner: HostBackend::Gpu { filter, render_params: &render_params },
			layer_width: w,
			layer_height: h,
			output_width: w,
			output_height: h,
			frame_index,
			time_seconds,
			progress: base_cfg.progress,
		};

		let frame_data = E::frame_data(ctx).map_err(|e| {
			log::error!("[adapter] frame_data failed: {e:?}");
			pr::Error::Fail
		})?;

		let mut base = Self::build_invocation(&props, &base_cfg, w, h, bpp)?;
		base.render_generation = frame_index as u64;

		run_graph(self.graph(), &frame_data, &base).map_err(|e| {
			log::error!("[adapter] graph execute failed: {e:?}");
			pr::Error::Fail
		})
	}
}
