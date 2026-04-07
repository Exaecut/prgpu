use crate::{
	PrRect,
	gpu::{frames_as_slice, gpu_bytes_per_pixels},
};
use after_effects::log;
use premiere::{self as pr, PixelFormat, Property};

#[derive(Clone)]
pub struct GPURenderProperties<'a> {
	pub progress: f32,
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
	/// Dereferences raw pointers (`filter.instance_ptr`, `frames`, `out_frame`) which must be valid, non-null, and properly aligned.
	/// `frames` must reference an array of at least `frame_count` valid `PPixHand` elements.
	/// Assumes all Premiere SDK suite calls return handles tied to the same GPU device and lifetime context.
	/// Caller must guarantee no aliasing or concurrent mutation of underlying frame data during execution.
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

		let frames = frames_as_slice(frames, frame_count)?;

		let outgoing = frames.first().copied().ok_or(pr::Error::Fail)?;
		let incoming = if is_transition {
			frames.get(1).copied().ok_or(pr::Error::Fail)?
		} else {
			std::ptr::null_mut()
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

		let properties = if !incoming.is_null() {
			incoming
		} else if !outgoing.is_null() {
			outgoing
		} else if !out_frame.is_null() {
			unsafe { *out_frame }
		} else {
			return Err(pr::Error::Fail);
		};

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

		let half_precision = pixel_format != pr::PixelFormat::GpuBgra4444_32f;
		let bounds: after_effects::Rect = after_effects::Rect::from(PrRect::from(filter.ppix_suite.bounds(properties).unwrap()));

		let width = bounds.width();
		let height = bounds.height();

		let (par_numerator, par_denominator) = match filter.ppix_suite.pixel_aspect_ratio(properties) {
			Ok((num, den)) => (num as i32, den as i32),
			Err(_) => {
				log::error!("Failed to get pixel aspect ratio for properties");
				return Err(pr::Error::InvalidParms);
			}
		};

		let field_type = match filter.ppix2_suite.field_order(properties) {
			Ok(field_type) => field_type,
			Err(_) => {
				log::error!("Failed to get field type for properties");
				return Err(pr::Error::InvalidParms);
			}
		};

		let output_frame = match filter
			.gpu_device_suite
			.create_gpu_ppix(gpu_index, pixel_format, width, height, par_numerator, par_denominator, field_type)
		{
			Ok(frame) => frame,
			Err(_) => {
				log::error!("Failed to create GPU PPix");
				return Err(pr::Error::InvalidParms);
			}
		};

		if output_frame.is_null() {
			log::error!("Output frame is null");
			return Err(pr::Error::Fail);
		}

		unsafe { *out_frame = output_frame; }

		let bytes_per_pixel = gpu_bytes_per_pixels(pixel_format);

		Ok(GPURenderProperties {
			progress,
			gpu_index,
			pixel_format,
			half_precision,
			bounds,
			output_frame,
			bytes_per_pixel,
			frames: (incoming, outgoing),
			filter,
		})
	}

	pub fn get_filter(&self) -> &premiere::GpuFilterData {
		self.filter
	}
}
