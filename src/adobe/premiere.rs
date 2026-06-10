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

use after_effects::log;
use premiere::{self as pr};

use crate::effect::frame_context::{FrameDataContext, HostBackend};
use crate::effect::host::{Host, RenderKind};
use crate::effect::{Effect, FrameBinding, InvocationBase, LicenseGate, PixelLayout};
use crate::gpu::pipeline;
use crate::gpu::render_properties::GPURenderProperties;
use crate::graph::{RenderGraph, execute::execute as run_graph};
use crate::types::{Backend, Configuration};

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

	fn build_invocation(props: &GPURenderProperties<'_>, base_cfg: &Configuration, bpp: u32) -> Result<InvocationBase, pr::Error> {
		let pixel_layout = PixelLayout::from_u32(base_cfg.pixel_layout);

		// Source carries the clip's own extent; output carries the canvas
		// (sequence size on the pure GPU-filter path). They differ when
		// Premiere hands a sequence-sized outFrame for a smaller clip.
		let main = FrameBinding {
			data: base_cfg.outgoing_data.unwrap_or(std::ptr::null_mut()),
			pitch_px: base_cfg.outgoing_pitch_px,
			width: base_cfg.layer_width,
			height: base_cfg.layer_height,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};
		let output = FrameBinding {
			data: base_cfg.dest_data,
			pitch_px: base_cfg.dest_pitch_px,
			width: base_cfg.canvas_width,
			height: base_cfg.canvas_height,
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
			storage: base_cfg.storage,
			flip_y: 0, // Premiere GPU PPix is top-down
			time: base_cfg.time,
			progress: base_cfg.progress,
			render_generation: base_cfg.render_generation,
			ext_x: base_cfg.ext_x,
			ext_y: base_cfg.ext_y,
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
			#[cfg(gpu_backend = "cuda")]
			crate::gpu::backends::cuda::frame_scope::cleanup();
		}
	}

	fn get_frame_dependencies(
		&self,
		_filter: &pr::GpuFilterData,
		_render_params: pr::RenderParams,
		_query_index: &mut i32,
	) -> Result<pr::sys::PrGPUFilterFrameDependency, pr::Error> {
		Err(pr::Error::NotImplemented)
	}

	fn precompute(&self, _filter: &pr::GpuFilterData, _render_params: pr::RenderParams, _index: i32, _frame: pr::sys::PPixHand) -> Result<(), pr::Error> {
		Ok(())
	}

	fn render(
		&self,
		filter: &pr::GpuFilterData,
		render_params: pr::RenderParams,
		frames: *const pr::sys::PPixHand,
		frame_count: usize,
		out_frame: *mut pr::sys::PPixHand,
	) -> Result<(), pr::Error> {
		if !self.license.is_valid() {
			return Ok(());
		}

		// Before the potentially-destructive canvas swap in GPURenderProperties,
		// call Effect::expansion() to decide whether the user requested expansion.
		// Only when the expansion extent is non-zero do we allocate a canvas PPix
		// (the WonderGlow pattern); otherwise the clip-sized in-place path stays.
		let expand_to_canvas = {
			// Safely read the first frame handle.
			let first = if !frames.is_null() {
				unsafe { Some(*frames) }
			} else {
				None
			};
			let clip = first.unwrap_or(std::ptr::null_mut());
			let clip_ppix = if !clip.is_null() {
				clip
			} else {
				unsafe { *out_frame }
			};
			let clip_w = if !clip_ppix.is_null() {
				filter.ppix_suite.row_bytes(clip_ppix).map(|rb| {
					let bpp = filter
						.ppix_suite
						.pixel_format(clip_ppix)
						.map(|pf| crate::gpu::gpu_bytes_per_pixels(pf))
						.unwrap_or(0);
					if bpp > 0 { (rb / bpp) as u32 } else { 0 }
				}).unwrap_or(0)
			} else {
				0
			};
			let clip_h = if !clip_ppix.is_null() {
				filter
					.gpu_device_suite
					.gpu_ppix_size(clip_ppix)
					.map(|s| {
						let rb = filter.ppix_suite.row_bytes(clip_ppix).unwrap_or(1);
						(s / rb as usize) as u32
					})
					.unwrap_or(0)
			} else {
				0
			};
			if clip_w == 0 || clip_h == 0 {
				false // Can't determine clip dims — don't expand.
			} else {
				let exp_ctx = crate::effect::frame_context::FrameDataContext {
					host: crate::effect::host::Host::Premiere,
					backend: {
						#[cfg(gpu_backend = "cuda")]
						{ crate::types::Backend::Cuda }
						#[cfg(gpu_backend = "metal")]
						{ crate::types::Backend::Metal }
						#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
						{ crate::types::Backend::Cpu }
					},
					render_kind: crate::effect::host::RenderKind::PremiereGpuEffect,
					inner: crate::effect::frame_context::HostBackend::Gpu {
						filter,
						render_params: &render_params,
					},
					layer_width: clip_w,
					layer_height: clip_h,
					output_width: clip_w,
					output_height: clip_h,
					ext_x: 0,
					ext_y: 0,
					frame_index: 0,
					time_seconds: 0.0,
					progress: 0.0,
				};
				match E::expansion(exp_ctx) {
					Ok(ext) => !ext.is_zero(),
					Err(_) => false,
				}
			}
		};

		let props = unsafe { GPURenderProperties::new(filter, render_params.clone(), frames, frame_count, out_frame, expand_to_canvas) }?;
		let base_cfg = unsafe { Configuration::effect(&props, out_frame)? };

		let w = base_cfg.width;
		let h = base_cfg.height;
		if w == 0 || h == 0 {
			return Ok(());
		}
		let bpp = props.bytes_per_pixel as u32;

		// Source pitch must cover the source width. Both now come from the same
		// native PPix (`GPURenderProperties` derives dims from the output buffer),
		// so this only trips on a genuinely malformed frame rather than on
		// legitimately small stills (a 400x400 image in a 1080p sequence).
		let expected_pitch_bytes = base_cfg.outgoing_width.saturating_mul(bpp);
		let src_pitch_bytes = (base_cfg.outgoing_pitch_px as u32).saturating_mul(bpp);
		if src_pitch_bytes < expected_pitch_bytes {
			log::warn!("[adapter] skipping frame: source pitch {src_pitch_bytes} < expected {expected_pitch_bytes}. dims={w}x{h} bpp={bpp}");
			return Ok(());
		}

		let frame_index = if render_params.render_ticks_per_frame() != 0 {
			(render_params.sequence_time() / render_params.render_ticks_per_frame()) as u32
		} else {
			0
		};
		// Canonical time already lives in base_cfg.time (sequence seconds, set by
		// Configuration::effect); reuse it so frame_data and the shader agree.
		let time_seconds = base_cfg.time;
		let quality = render_params.quality();

		// Dispatch-boundary evidence. `storage` is what vekl will use to decode the
		// buffer; `frame.time` is the value that actually reaches the shader (distinct
		// from the seconds value handed to `frame_data`).
		let storage_tag = base_cfg.storage;
		let storage_label = match storage_tag {
			0 => "Unorm8x4",
			1 => "Unorm16x4",
			2 => "Float32x4",
			3 => "Float16x4",
			_ => "?",
		};

		log::info!(
			"[GPU] frame {frame_index} t_sec={time_seconds:.4} frame.time={frame_time} canvas={w}x{h} layer={lw}x{lh} ext=({ex},{ey}) bpp={bpp} storage={storage_tag}({storage_label}) layout={layout}(0=RGBA,1=BGRA) pixel_format={pf:?} half_precision={half} src_pitch_px={src_pitch} dst_pitch_px={dst_pitch} seq_time={seq} clip_time={clip} ticks_per_frame={tpf} quality={quality:?}",
			frame_time = base_cfg.time,
			lw = base_cfg.layer_width,
			lh = base_cfg.layer_height,
			ex = base_cfg.ext_x,
			ey = base_cfg.ext_y,
			layout = base_cfg.pixel_layout,
			pf = props.pixel_format,
			half = props.half_precision,
			src_pitch = base_cfg.outgoing_pitch_px,
			dst_pitch = base_cfg.dest_pitch_px,
			seq = render_params.sequence_time(),
			clip = render_params.clip_time(),
			tpf = render_params.render_ticks_per_frame(),
		);

		// Regression guard: a half-float host format must resolve to Float16x4.
		if props.half_precision && storage_tag != crate::types::PIXEL_STORAGE_FLOAT16X4 {
			log::warn!(
				"[GPU] STORAGE REGRESSION: {pf:?} is half-float (16f) but storage tag is {storage_tag} (expected Float16x4=3); decode will be wrong.",
				pf = props.pixel_format,
			);
		}

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
			inner: HostBackend::Gpu {
				filter,
				render_params: &render_params,
			},
			layer_width: base_cfg.layer_width,
			layer_height: base_cfg.layer_height,
			output_width: base_cfg.canvas_width,
			output_height: base_cfg.canvas_height,
			ext_x: base_cfg.ext_x,
			ext_y: base_cfg.ext_y,
			frame_index,
			time_seconds,
			progress: base_cfg.progress,
		};

		let frame_data = E::frame_data(ctx).map_err(|e| {
			log::error!("[adapter] frame_data failed: {e:?}");
			pr::Error::Fail
		})?;

		let mut base = Self::build_invocation(&props, &base_cfg, bpp)?;
		base.render_generation = frame_index as u64;

		// One frame scope: backends enqueue every pass without per-pass syncs;
		// the single sync in `end` satisfies Adobe's "inputs consumed before
		// render() returns" contract. The macOS GPU watchdog can kill the
		// frame-level command buffer on a cold first dispatch — retry the
		// whole frame once before failing.
		use crate::gpu::frame_scope;
		let scope_desc = crate::types::FrameScopeDesc::from_invocation(&base);
		const MAX_FRAME_ATTEMPTS: u32 = 2;
		for attempt in 1..=MAX_FRAME_ATTEMPTS {
			frame_scope::begin(&scope_desc);

			let result = run_graph(self.graph(), &frame_data, &base);
			let sync = frame_scope::end(&scope_desc);

			if let Err(e) = result {
				log::error!("[adapter] graph execute failed: {e:?}");
				return Err(pr::Error::Fail);
			}
			match sync {
				Ok(()) => return Ok(()),
				Err(e) if e == frame_scope::ERR_WATCHDOG && attempt < MAX_FRAME_ATTEMPTS => {
					log::warn!("[adapter] frame hit GPU watchdog (attempt {attempt}/{MAX_FRAME_ATTEMPTS}) — cooling down 50ms and retrying");
					std::thread::sleep(std::time::Duration::from_millis(50));
				}
				Err(e) => {
					log::error!("[adapter] frame sync failed: {e}");
					return Err(pr::Error::Fail);
				}
			}
		}
		Err(pr::Error::Fail)
	}
}
