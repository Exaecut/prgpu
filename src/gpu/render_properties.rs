use crate::gpu::{frames_as_slice, gpu_bytes_per_pixels};
use after_effects::log;
use premiere::{self as pr, PixelFormat, Property};

#[derive(Clone)]
pub struct GPURenderProperties<'a> {
	pub progress: f32,
	pub time: f32,
	pub gpu_index: u32,
	pub pixel_format: PixelFormat,
	pub half_precision: bool,
	pub bounds: after_effects::Rect,
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

		let output_frame = unsafe { *out_frame };
		if output_frame.is_null() {
			log::error!("Output frame is null");
			return Err(pr::Error::Fail);
		}

		let half_precision = pixel_format != pr::PixelFormat::GpuBgra4444_32f;

		let bytes_per_pixel = gpu_bytes_per_pixels(pixel_format);

		// Premiere never expands frames: the source and output PPix are always at
		// the clip's native resolution, which can differ from the sequence size
		// `render_params.render_*()` reports (e.g. a 400x400 still in a 1920x1080
		// sequence). Derive the real dims from the output buffer — `row_bytes` is
		// the tight pixel pitch (width) and `gpu_ppix_size / row_bytes` is the
		// height. The PPix `bounds()` rect is unreliable on some Metal GPU PPix, so
		// the sequence size is only a fallback when the buffer query fails.
		let buffer_bounds = {
			let row_bytes = filter.ppix_suite.row_bytes(output_frame).unwrap_or(0);
			let size = filter.gpu_device_suite.gpu_ppix_size(output_frame).unwrap_or(0);
			if row_bytes > 0 && bytes_per_pixel > 0 && size > 0 {
				let w = row_bytes / bytes_per_pixel;
				let h = (size / row_bytes as usize) as i32;
				(w > 0 && h > 0).then_some(after_effects::Rect { left: 0, top: 0, right: w, bottom: h })
			} else {
				None
			}
		};
		let bounds = buffer_bounds.unwrap_or_else(|| {
			let rw = render_params.render_width() as i32;
			let rh = render_params.render_height() as i32;
			after_effects::Rect { left: 0, top: 0, right: rw, bottom: rh }
		});

		let ticks_per_frame = render_params.render_ticks_per_frame();
		let time = if ticks_per_frame != 0 {
			render_params.clip_time() as f32 / ticks_per_frame as f32
		} else {
			0.0
		};

		Ok(GPURenderProperties {
			progress,
			time,
			gpu_index,
			pixel_format,
			half_precision,
			bounds,
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
