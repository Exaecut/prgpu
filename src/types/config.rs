use std::ffi::c_void;

use premiere::suites::GPUDevice;

use crate::gpu::scheduling;
use crate::render_properties::GPURenderProperties;

pub enum DeviceHandleInit<'a> {
	FromPtr(*mut c_void),
	FromSuite((u32, &'a GPUDevice)),
}

#[repr(C)]
pub struct MTLSize {
	pub width: usize,
	pub height: usize,
	pub depth: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
pub struct Configuration {
	pub device_handle: *mut c_void,
	pub context_handle: Option<*mut c_void>,
	pub command_queue_handle: *mut c_void,
	pub outgoing_data: Option<*mut c_void>,
	pub incoming_data: Option<*mut c_void>,
	pub dest_data: *mut c_void,
	pub outgoing_pitch_px: i32,
	pub incoming_pitch_px: i32,
	pub dest_pitch_px: i32,
	pub width: u32,
	pub height: u32,
	pub is16f: bool,
	pub bytes_per_pixel: u32,
	pub progress: f32,
	pub render_generation: u64,
	pub pixel_layout: u32, // 0=RGBA, 1=BGRA, 2=VUYA601, 3=VUYA709
}

impl Configuration {
	/// # Safety
	/// `out_frame` must be a valid, non-null GPU frame pointer whose memory remains alive and writable.
	/// `bytes_per_pixel` and `row_bytes` must match the actual pixel format and layout.
	/// No concurrent access or invalid GPU context usage is allowed.
	pub unsafe fn effect(render_properties: &GPURenderProperties, out_frame: *mut premiere::sys::PPixHand) -> Result<Self, premiere::Error> {
		let filter = render_properties.get_filter();
		let bytes_per_pixel = render_properties.bytes_per_pixel;

		let (incoming, outgoing) = render_properties.frames;

		let (outgoing_data, outgoing_pitch_px) = if !outgoing.is_null() {
			let data = filter.gpu_device_suite.gpu_ppix_data(outgoing)?;
			let row_bytes = filter.ppix_suite.row_bytes(outgoing)?;
			(Some(data), row_bytes / bytes_per_pixel)
		} else {
			(None, 0)
		};

		let (incoming_data, incoming_pitch_px) = if !incoming.is_null() {
			let data = filter.gpu_device_suite.gpu_ppix_data(incoming)?;
			let row_bytes = filter.ppix_suite.row_bytes(incoming)?;
			(Some(data), row_bytes / bytes_per_pixel)
		} else {
			(None, 0)
		};

		let (dest_data, dest_row_bytes) = (
			filter.gpu_device_suite.gpu_ppix_data(unsafe { *out_frame })?,
			filter.ppix_suite.row_bytes(unsafe { *out_frame })?,
		);
		let dest_pitch_px = dest_row_bytes / bytes_per_pixel;

		let width = render_properties.bounds.width();
		let height = render_properties.bounds.height();

		Ok(Self {
			device_handle: filter.gpu_info.outDeviceHandle,
			context_handle: Some(filter.gpu_info.outContextHandle),
			command_queue_handle: filter.gpu_info.outCommandQueueHandle,
			outgoing_data,
			incoming_data,
			dest_data,
			outgoing_pitch_px,
			incoming_pitch_px,
			dest_pitch_px,
			width: width as u32,
			height: height as u32,
			is16f: render_properties.half_precision,
			bytes_per_pixel: render_properties.bytes_per_pixel as u32,
			progress: render_properties.progress,
			render_generation: scheduling::advance_generation(),
			pixel_layout: 1, // GPU path always receives BGRA from Premiere
		})
	}

	/// Builds a `Configuration` for CPU (After Effects software render).
	///
	/// `in_data` and `out_data` must point to valid pixel buffers for the
	/// duration of the kernel dispatch. Pitches are in pixels, not bytes.
	pub fn cpu(in_data: *mut c_void, out_data: *mut c_void, in_pitch_px: i32, out_pitch_px: i32, width: u32, height: u32, is16f: bool, bytes_per_pixel: u32, pixel_layout: u32) -> Self {
		Self {
			device_handle: std::ptr::null_mut(),
			context_handle: None,
			command_queue_handle: std::ptr::null_mut(),
			outgoing_data: Some(in_data),
			incoming_data: Some(in_data),
			dest_data: out_data,
			outgoing_pitch_px: in_pitch_px,
			incoming_pitch_px: in_pitch_px,
			dest_pitch_px: out_pitch_px,
			width,
			height,
			is16f,
			bytes_per_pixel,
			progress: 0.0,
			render_generation: 0,
			pixel_layout,
		}
	}

	/// # Safety
	/// `out_frame` must be a valid, non-null GPU frame pointer whose memory remains alive and writable.
	/// `bytes_per_pixel` and `row_bytes` must match the actual pixel format and layout.
	/// No concurrent access or invalid GPU context usage is allowed.
	pub unsafe fn transition(render_properties: &GPURenderProperties, out_frame: *mut premiere::sys::PPixHand) -> Result<Self, premiere::Error> {
		let filter = render_properties.get_filter();
		let bytes_per_pixel = render_properties.bytes_per_pixel;

		let (incoming, outgoing) = render_properties.frames;

		let (incoming_data, incoming_row_bytes) = (Some(filter.gpu_device_suite.gpu_ppix_data(incoming)?), filter.ppix_suite.row_bytes(incoming)?);
		let incoming_pitch_px = incoming_row_bytes / bytes_per_pixel;

		let (outgoing_data, outgoing_row_bytes) = (Some(filter.gpu_device_suite.gpu_ppix_data(outgoing)?), filter.ppix_suite.row_bytes(outgoing)?);
		let outgoing_pitch_px = outgoing_row_bytes / bytes_per_pixel;

		let (dest_data, dest_row_bytes) = (
			filter.gpu_device_suite.gpu_ppix_data(unsafe { *out_frame })?,
			filter.ppix_suite.row_bytes(unsafe { *out_frame })?,
		);

		let dest_pitch_px = dest_row_bytes / bytes_per_pixel;

		let width = render_properties.bounds.width();
		let height = render_properties.bounds.height();

		Ok(Self {
			device_handle: filter.gpu_info.outDeviceHandle,
			context_handle: Some(filter.gpu_info.outContextHandle),
			command_queue_handle: filter.gpu_info.outCommandQueueHandle,
			outgoing_data,
			incoming_data,
			dest_data,
			outgoing_pitch_px,
			incoming_pitch_px,
			dest_pitch_px,
			width: width as u32,
			height: height as u32,
			is16f: render_properties.half_precision,
			bytes_per_pixel: render_properties.bytes_per_pixel as u32,
			progress: render_properties.progress,
			render_generation: scheduling::advance_generation(),
			pixel_layout: 1, // GPU path always receives BGRA from Premiere
		})
	}
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameParams {
	pub out_pitch: u32,
	pub in_pitch: u32,
	pub dest_pitch: u32,
	pub width: u32,
	pub height: u32,
	pub progress: f32,
	pub bpp: u32,
	pub pixel_layout: u32, // 0=RGBA, 1=BGRA, 2=VUYA601, 3=VUYA709
}
