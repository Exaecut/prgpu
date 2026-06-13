//! Premiere GPU adapter: drives `pr::GpuFilter` through the [`Effect`] trait.
//!
//! Effects declare:
//!
//! ```ignore
//! pub type PremiereGPU = prgpu::adobe::premiere::GpuFilterAdapter<MyEffect, prgpu::NoLicense>;
//! pr::define_gpu_filter!(PremiereGPU);
//! ```
//!
//! The adapter snapshots parameters via `E::Params::snapshot_gpu()`, builds a
//! `Ctx<E::Params>`, resolves expansion, then runs the cached `Graph<E::Params>`.

use std::sync::OnceLock;

use after_effects::log;
use premiere::{self as pr};

use crate::effect::ctx::{Ctx, Geometry, Timing};
use crate::effect::host::{Host, HostCapabilities, RenderKind};
use crate::effect::{Effect, FrameBinding, InvocationBase, LicenseGate, PixelLayout};
use crate::gpu::pipeline;
use crate::gpu::render_properties::GPURenderProperties;
use crate::graph::{Graph, execute::execute as run_graph};
use crate::params::{ParamsSpec, SnapshotGeom};
use crate::types::{Backend, Configuration, FrameScopeDesc};

pub struct GpuFilterAdapter<E: Effect, L: LicenseGate> {
	license: L,
	graph: OnceLock<Graph<E::Params>>,
}

impl<E: Effect, L: LicenseGate> Default for GpuFilterAdapter<E, L> {
	fn default() -> Self {
		Self {
			license: L::default(),
			graph: OnceLock::new(),
		}
	}
}

impl<E: Effect, L: LicenseGate> GpuFilterAdapter<E, L> {
	fn graph(&self) -> &Graph<E::Params> {
		self.graph.get_or_init(|| {
			let mut g = Graph::new();
			E::pipeline(&mut g);
			g
		})
	}

	#[inline]
	fn license_valid(&self) -> bool {
		let ok = self.license.is_valid();
		#[cfg(debug_assertions)]
		if !ok {
			after_effects::log::warn!("license: gate closed, render skipped; state=[{}]", self.license.debug_label().unwrap_or_default());
		}
		ok
	}

	fn build_invocation(props: &GPURenderProperties<'_>, base_cfg: &Configuration, bpp: u32) -> Result<InvocationBase, pr::Error> {
		let pixel_layout = PixelLayout::from_u32(base_cfg.pixel_layout);

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

		#[cfg(gpu_backend = "cuda")]
		let device_handle = base_cfg.context_handle.unwrap_or(std::ptr::null_mut());
		#[cfg(gpu_backend = "metal")]
		let device_handle = base_cfg.device_handle;
		#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
		let device_handle: *mut std::ffi::c_void = std::ptr::null_mut();

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
			flip_y: 0,
			time: base_cfg.time,
			progress: base_cfg.progress,
			render_generation: base_cfg.render_generation,
			ext_x: base_cfg.ext_x,
			ext_y: base_cfg.ext_y,
			source: main,
			secondary_source: None,
			output,
		})
	}

	fn backend() -> Backend {
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

	fn expand_to_canvas(filter: &pr::GpuFilterData, render_params: &pr::RenderParams, frames: *const pr::sys::PPixHand, out_frame: *mut pr::sys::PPixHand) -> bool {
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
			return false;
		}

		let frame_index = if render_params.render_ticks_per_frame() != 0 {
			(render_params.sequence_time() / render_params.render_ticks_per_frame()) as u32
		} else {
			0
		};
		let time_seconds = crate::adobe::ticks_to_seconds(render_params.sequence_time());

		let geom = SnapshotGeom {
			layer_w: clip_w,
			layer_h: clip_h,
			output_w: clip_w,
			output_h: clip_h,
			ext_x: 0,
			ext_y: 0,
		};
		let snapshot = E::Params::snapshot_gpu(filter, render_params, &geom);
		let ctx = Ctx::new(
			&snapshot,
			Geometry { layer_w: clip_w, layer_h: clip_h, output_w: clip_w, output_h: clip_h, ext_x: 0, ext_y: 0 },
			Timing { frame_index, time_seconds, progress: 0.0 },
			HostCapabilities::new(Host::Premiere, Self::backend()),
			false,
		);

		!E::expansion(&ctx).is_zero()
	}
}

impl<E: Effect, L: LicenseGate> pr::GpuFilter for GpuFilterAdapter<E, L> {
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
		if !self.license_valid() {
			return Ok(());
		}

		let expand_to_canvas = Self::expand_to_canvas(filter, &render_params, frames, out_frame);

		let props = unsafe { GPURenderProperties::new(filter, render_params.clone(), frames, frame_count, out_frame, expand_to_canvas) }?;
		let base_cfg = unsafe { Configuration::effect(&props, out_frame)? };

		let w = base_cfg.width;
		let h = base_cfg.height;
		if w == 0 || h == 0 {
			return Ok(());
		}
		let bpp = props.bytes_per_pixel as u32;

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
		let time_seconds = base_cfg.time;
		let quality = render_params.quality();

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

		if props.half_precision && storage_tag != crate::types::PIXEL_STORAGE_FLOAT16X4 {
			log::warn!(
				"[GPU] STORAGE REGRESSION: {pf:?} is half-float (16f) but storage tag is {storage_tag} (expected Float16x4=3); decode will be wrong.",
				pf = props.pixel_format,
			);
		}

		let geom = SnapshotGeom {
			layer_w: base_cfg.layer_width,
			layer_h: base_cfg.layer_height,
			output_w: base_cfg.canvas_width,
			output_h: base_cfg.canvas_height,
			ext_x: base_cfg.ext_x,
			ext_y: base_cfg.ext_y,
		};
		let snapshot = E::Params::snapshot_gpu(filter, &render_params, &geom);

		let backend = Self::backend();
		let debug_view = false;
		let ctx = Ctx::new(
			&snapshot,
			Geometry {
				layer_w: base_cfg.layer_width,
				layer_h: base_cfg.layer_height,
				output_w: base_cfg.canvas_width,
				output_h: base_cfg.canvas_height,
				ext_x: base_cfg.ext_x,
				ext_y: base_cfg.ext_y,
			},
			Timing { frame_index, time_seconds, progress: base_cfg.progress },
			HostCapabilities::new(Host::Premiere, backend),
			debug_view,
		);

		let mut base = Self::build_invocation(&props, &base_cfg, bpp)?;
		base.render_generation = frame_index as u64;

		use crate::gpu::frame_scope;
		let scope_desc = FrameScopeDesc::from_invocation(&base);
		const MAX_FRAME_ATTEMPTS: u32 = 2;
		for attempt in 1..=MAX_FRAME_ATTEMPTS {
			frame_scope::begin(&scope_desc);

			let result = run_graph(self.graph(), &ctx, &base);
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
