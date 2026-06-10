use crate::gpu::{frames_as_slice, gpu_bytes_per_pixels, gpu_storage};
use after_effects::log;
use premiere::{self as pr, PixelFormat, Property};

#[derive(Clone)]
pub struct GPURenderProperties<'a> {
	pub progress: f32,
	pub time: f32,
	pub gpu_index: u32,
	pub pixel_format: PixelFormat,
	pub half_precision: bool,
	pub storage: u32,
	/// Output canvas. On the Premiere GPU-filter path this is the `outFrame`
	/// PPix extent — the sequence resolution once the AE-GPU flags are dropped
	pub bounds: after_effects::Rect,
	/// Un-expanded source clip extent from `frames[0]`'s PPix. Differs from
	/// `bounds` when Premiere hands a sequence-sized outFrame for a smaller clip.
	pub layer_bounds: after_effects::Rect,
	/// Source clip top-left inside the canvas (0,0 when not expanded), derived
	/// from the input PPix origin.
	pub ext_x: i32,
	pub ext_y: i32,
	pub output_frame: pr::sys::PPixHand,
	pub frames: (pr::sys::PPixHand, pr::sys::PPixHand),
	pub bytes_per_pixel: i32,

	filter: &'a premiere::GpuFilterData,
}

impl<'a> GPURenderProperties<'a> {
	/// # Safety
	/// `filter.instance_ptr`, `frames`, `out_frame` must be valid, non-null, and aligned;
	/// `frames` must hold at least `frame_count` valid `PPixHand`s. No aliasing or
	/// concurrent mutation of frame data.
	pub unsafe fn new(
		filter: &'a premiere::GpuFilterData,
		render_params: premiere::RenderParams,
		frames: *const premiere::sys::PPixHand,
		frame_count: usize,
		out_frame: *mut premiere::sys::PPixHand,
		// When true AND the host output is smaller than the render canvas,
		// allocate a canvas-sized GPU PPix and swap it into *out_frame
		// (WonderGlow FUN_1800d9ab0 pattern). Set by the adapter after
		// calling Effect::expansion() — only gated on expansion != none().
		expand_to_canvas: bool,
	) -> Result<Self, premiere::Error> {
		assert!(!out_frame.is_null(), "out_frame pointer must not be null");

		unsafe {
			(*filter.instance_ptr).outIsRealtime = 1;
		}

		let is_transition = frame_count >= 2 && pr::suites::Transition::new().is_ok();

		let raw_frames = frames_as_slice(frames, frame_count).unwrap_or(&[]);
		let first = raw_frames.first().copied().unwrap_or(std::ptr::null_mut());

		let outgoing = if !first.is_null() { first } else { std::ptr::null_mut() };
		let incoming = if is_transition {
			raw_frames.get(1).copied().unwrap_or(std::ptr::null_mut())
		} else {
			std::ptr::null_mut()
		};

		// Use the output frame as source if the first input frame is missing.
		let main_source = if !outgoing.is_null() {
			outgoing
		} else if !out_frame.is_null() {
			unsafe { *out_frame }
		} else {
			return Err(pr::Error::Fail);
		};

		let key = if is_transition {
			Property::Transition_Duration
		} else {
			Property::Effect_EffectDuration
		};

		let progress = match filter.property(key) {
			Ok(pr::PropertyData::Int64(d)) if d != 0 => render_params.clip_time() as f64 / d as f64,
			Ok(pr::PropertyData::Time(d)) if d != 0 => render_params.clip_time() as f64 / d as f64,
			Ok(property_data) => {
				log::error!("Retrieved unexpected property data: {property_data:?}");
				return Err(pr::Error::InvalidParms);
			}
			Err(error) => {
				log::error!("Failed to get transition duration: {error:?}");
				return Err(pr::Error::InvalidParms);
			}
		} as f32;

		// Prefer a source that actually has GPU data.
		let mut source = if !incoming.is_null() { incoming } else { main_source };
		if filter.gpu_device_suite.gpu_ppix_data(source).is_err() {
			if !out_frame.is_null() {
				let out = unsafe { *out_frame };
				if filter.gpu_device_suite.gpu_ppix_data(out).is_ok() {
					source = out;
				}
			}
		}

		let properties = source;
		let gpu_index = match filter.gpu_device_suite.gpu_ppix_device_index(properties) {
			Ok(index) => index,
			Err(_) => {
				log::error!("Failed to get GPU device index for properties");
				return Err(pr::Error::InvalidParms);
			}
		};

		let pixel_format = match filter.ppix_suite.pixel_format(properties) {
			Ok(format) => format,
			Err(_) => {
				log::error!("Failed to get pixel format for properties");
				return Err(pr::Error::InvalidParms);
			}
		};

		let mut output_frame = unsafe { *out_frame };
		if output_frame.is_null() {
			log::error!("Output frame is null");
			return Err(pr::Error::Fail);
		}

		let half_precision = pixel_format != pr::PixelFormat::GpuBgra4444_32f;
		let storage = gpu_storage(pixel_format);

		let bytes_per_pixel = gpu_bytes_per_pixels(pixel_format);

		// Content extent of a PPix. `bounds()` gives the true pixel rect; the
		// buffer geometry (`row_bytes / bpp` × `gpu_ppix_size / row_bytes`) is
		// pitch-padded capacity, NOT content size — using it as width leaks the
		// row padding into the visible canvas (right-edge garbage band when
		// expanded). Buffer capacity is kept only as an upper-bound sanity check
		// and as fallback for PPix where bounds() is unreliable (some Metal GPU
		// PPix return empty rects).
		let ppix_extent = |frame: pr::sys::PPixHand| -> Option<after_effects::Rect> {
			let row_bytes = filter.ppix_suite.row_bytes(frame).unwrap_or(0);
			let size = filter.gpu_device_suite.gpu_ppix_size(frame).unwrap_or(0);
			let capacity = if row_bytes > 0 && bytes_per_pixel > 0 && size > 0 {
				let w = row_bytes / bytes_per_pixel;
				let h = (size / row_bytes as usize) as i32;
				(w > 0 && h > 0).then_some((w, h))
			} else {
				None
			};

			if let Ok(r) = filter.ppix_suite.bounds(frame) {
				let w = r.right - r.left;
				let h = r.bottom - r.top;
				let fits = capacity.map(|(cw, ch)| w <= cw && h <= ch).unwrap_or(true);
				if w > 0 && h > 0 && fits {
					return Some(after_effects::Rect { left: 0, top: 0, right: w, bottom: h });
				}
			}

			capacity.map(|(w, h)| after_effects::Rect { left: 0, top: 0, right: w, bottom: h })
		};

		// Premiere hands a GPU filter the clip-sized frame as the in-place output
		// (outFrame may share the same handle as frames[0]) and reports the
		// sequence size via render_*. When the clip is smaller than the canvas,
		// mirror WonderGlow FUN_1800d9ab0: allocate a canvas-sized GPU PPix and
		// swap it into *out_frame. The source always stays the (smaller) input
		// PPix and is sampled at `ext`. The kernel's early-return contract
		// limits writes to the expansion extent around `ext`.
		let render_w = render_params.render_width() as i32;
		let render_h = render_params.render_height() as i32;
		let canvas = after_effects::Rect { left: 0, top: 0, right: render_w, bottom: render_h };
		let out_extent = ppix_extent(output_frame);
		let layer_extent = if !outgoing.is_null() { ppix_extent(outgoing) } else { None };

		let host_out_w = out_extent.map(|r| r.width()).unwrap_or(render_w);
		let host_out_h = out_extent.map(|r| r.height()).unwrap_or(render_h);
		// Expand when the effect requests it (expansion extent != none) AND the
		// host-provided output is strictly smaller than the sequence canvas.
		let needs_expansion = expand_to_canvas && (host_out_w < render_w || host_out_h < render_h);

		let mut bounds = out_extent.unwrap_or(canvas);
		let mut expanded = false;
		if needs_expansion {
			let (par_num, par_den) = render_params.render_pixel_aspect_ratio();
			let field = render_params.render_field_type();
			match filter
				.gpu_device_suite
				.create_gpu_ppix(gpu_index, pixel_format, render_w, render_h, par_num as i32, par_den as i32, field)
			{
				Ok(canvas_ppix) => {
					unsafe { *out_frame = canvas_ppix };
					output_frame = canvas_ppix;
					bounds = canvas;
					expanded = true;
				}
				Err(e) => {
					log::warn!("[GPU/props] create_gpu_ppix({render_w}x{render_h}) failed: {e:?}; rendering in place at {host_out_w}x{host_out_h}");
				}
			}
		}

		// Layer = the input clip's extent, read from the source handle (which may
		// alias the pre-swap output handle). Always distinct from the canvas when
		// expanded; collapses to the in-place canvas otherwise.
		let layer_bounds = layer_extent.or(out_extent).unwrap_or(bounds);

		let (ext_x, ext_y) = if expanded {
			match filter.ppix2_suite.origin(outgoing) {
				Ok((ox, oy)) => (-ox, -oy),
				Err(_) => (((render_w - layer_bounds.width()) / 2).max(0), ((render_h - layer_bounds.height()) / 2).max(0)),
			}
		} else {
			(0, 0)
		};

		log::info!(
			"[GPU/props] canvas={cw}x{ch} renderWH={render_w}x{render_h} host_out={host_out_w}x{host_out_h} layer={lw}x{lh} ext=({ext_x},{ext_y}) expanded={expanded} full_canvas={fc} src_pitch={sp} dest_pitch={dp}",
			cw = bounds.width(),
			ch = bounds.height(),
			lw = layer_bounds.width(),
			lh = layer_bounds.height(),
			sp = { let rb = filter.ppix_suite.row_bytes(outgoing).unwrap_or(0); let bpp = bytes_per_pixel; if bpp > 0 { rb / bpp } else { 0 } },
			dp = { let rb = filter.ppix_suite.row_bytes(output_frame).unwrap_or(0); let bpp = bytes_per_pixel; if bpp > 0 { rb / bpp } else { 0 } },
			fc = bounds.width() == render_w && bounds.height() == render_h,
		);

		// Canonical effect time: sequence/timeline seconds, matching the CPU path
		// (PF_UtilitySuite::GetSequenceTime). frame.time is seconds on every backend.
		let time = crate::adobe::ticks_to_seconds(render_params.sequence_time());

		Ok(GPURenderProperties {
			progress,
			time,
			gpu_index,
			pixel_format,
			half_precision,
			storage,
			bounds,
			layer_bounds,
			ext_x,
			ext_y,
			output_frame,
			bytes_per_pixel,
			frames: (incoming, source),
			filter,
		})
	}

	pub fn get_filter(&self) -> &premiere::GpuFilterData {
		self.filter
	}
}
