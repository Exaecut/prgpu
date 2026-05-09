use std::ffi::c_void;

use premiere::suites::GPUDevice;

use crate::gpu::scheduling;
use crate::render_properties::GPURenderProperties;

pub enum DeviceHandleInit<'a> {
	FromPtr(*mut c_void),
	FromSuite((u32, &'a GPUDevice)),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
	// `width`/`height` are DESTINATION dims (drive dispatch grid + dst_desc + frame.width/height).
	// `*_width`/`*_height` describe the actual size of the source buffers, which may differ from
	// the destination (e.g. multi-pass blur reading a downsampled intermediate).
	pub width: u32,
	pub height: u32,
	pub outgoing_width: u32,
	pub outgoing_height: u32,
	pub incoming_width: u32,
	pub incoming_height: u32,
	pub bytes_per_pixel: u32,
	pub time: f32,
	pub progress: f32,
	pub render_generation: u64,
	pub pixel_layout: u32, // 0=RGBA, 1=BGRA, 2=VUYA601, 3=VUYA709
	/// Number of mip levels to allocate and auto-generate on the outgoing
	/// (source) buffer, including level 0. `0` or `1` disables mip support
	/// entirely; `2..=MAX_MIP` requests an N-level pyramid. The effect
	/// kernel sees the populated `TextureDesc.mip_*` fields and can call
	/// `SampleLinear(uv, lod)` / `SampleLinearTrilinear(uv, lodF)`.
	pub outgoing_mip_levels: u32,
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
			outgoing_width: width as u32,
			outgoing_height: height as u32,
			incoming_width: width as u32,
			incoming_height: height as u32,
			bytes_per_pixel: render_properties.bytes_per_pixel as u32,
			time: render_properties.time,
			progress: render_properties.progress,
			render_generation: scheduling::advance_generation(),
			pixel_layout: 1, // GPU path always receives BGRA from Premiere
			outgoing_mip_levels: 0,
		})
	}

	pub fn cpu(in_data: *mut c_void, out_data: *mut c_void, in_pitch_px: i32, out_pitch_px: i32, width: u32, height: u32, bytes_per_pixel: u32, pixel_layout: u32) -> Self {
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
			outgoing_width: width,
			outgoing_height: height,
			incoming_width: width,
			incoming_height: height,
			bytes_per_pixel,
			time: 0.0,
			progress: 0.0,
			render_generation: 0,
			pixel_layout,
			outgoing_mip_levels: 0,
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
			outgoing_width: width as u32,
			outgoing_height: height as u32,
			incoming_width: width as u32,
			incoming_height: height as u32,
			bytes_per_pixel: render_properties.bytes_per_pixel as u32,
			time: render_properties.time,
			progress: render_properties.progress,
			render_generation: scheduling::advance_generation(),
			pixel_layout: 1, // GPU path always receives BGRA from Premiere
			outgoing_mip_levels: 0,
		})
	}
}

/// Upper bound on mip levels tracked by `TextureDesc`. Must match
/// `vekl::MAX_MIP` or the ConstantBuffer layout will be mismatched.
/// Five levels cover down to 1/16 per axis — deep enough for any sweep
/// blur pyramid and keeps the descriptor small.
pub const MAX_MIP: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TextureDesc {
	pub width: u32,
	pub height: u32,
	pub pitch_bytes: u32,
	pub bytes_per_pixel: u32,
	pub storage: u32,
	pub layout: u32,
	pub address_mode: u32,

	// Mip-chain metadata. `mip_level_count >= 1`; entries beyond that are
	// undefined. The slang side uses `uint[MAX_MIP]` to match this layout
	// byte-for-byte.
	pub mip_level_count: u32,
	pub mip_offset_bytes: [u32; MAX_MIP as usize],
	pub mip_width: [u32; MAX_MIP as usize],
	pub mip_height: [u32; MAX_MIP as usize],
	pub mip_pitch_bytes: [u32; MAX_MIP as usize],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameParams {
	pub out_desc: TextureDesc,
	pub in_desc: TextureDesc,
	pub dst_desc: TextureDesc,
	pub width: u32,
	pub height: u32,
	pub time: f32,
	pub progress: f32,
}

pub const PIXEL_STORAGE_UNORM8X4: u32 = 0;
pub const PIXEL_STORAGE_UNORM16X4: u32 = 1;
pub const PIXEL_STORAGE_FLOAT32X4: u32 = 2;

pub fn storage_from_bpp(bpp: u32) -> u32 {
	match bpp {
		8 => PIXEL_STORAGE_UNORM16X4,
		16 => PIXEL_STORAGE_FLOAT32X4,
		_ => PIXEL_STORAGE_UNORM8X4,
	}
}

pub fn make_texture_desc(width: u32, height: u32, pitch_px: u32, bpp: u32, pixel_layout: u32) -> TextureDesc {
	let mut desc = TextureDesc {
		width,
		height,
		pitch_bytes: pitch_px * bpp,
		bytes_per_pixel: bpp,
		storage: storage_from_bpp(bpp),
		layout: pixel_layout,
		address_mode: 0, // Clamp
		mip_level_count: 1,
		mip_offset_bytes: [0; MAX_MIP as usize],
		mip_width: [0; MAX_MIP as usize],
		mip_height: [0; MAX_MIP as usize],
		mip_pitch_bytes: [0; MAX_MIP as usize],
	};
	// Level 0 always mirrors the base dims, so kernels that only touch
	// `Size(0)`/`Load(px, 0)` see a fully-filled descriptor even when
	// the host never explicitly requested a mip chain.
	desc.mip_width[0] = width;
	desc.mip_height[0] = height;
	desc.mip_pitch_bytes[0] = pitch_px * bpp;
	desc
}

/// Total byte size of a tightly packed mip chain (`levels` levels starting
/// at `width x height`). Each level's pitch is `width * bpp` (no padding);
/// level 0 matches the host's chosen pitch iff the host passes the same
/// width. Call [`fill_mip_desc`] to populate `TextureDesc` consistently.
pub fn mip_buffer_size_bytes(width: u32, height: u32, bpp: u32, levels: u32) -> u32 {
	let mut total = 0u32;
	let n = levels.max(1).min(MAX_MIP);
	for lvl in 0..n {
		let w = (width >> lvl).max(1);
		let h = (height >> lvl).max(1);
		total = total.saturating_add(w * h * bpp);
	}
	total
}

/// Build the outgoing-side [`TextureDesc`] from a [`Configuration`]. If the
/// caller requested a mip chain (`outgoing_mip_levels > 1`), the returned
/// descriptor's mip fields are populated via [`fill_mip_desc`]; otherwise the
/// descriptor is a plain single-level view. Called from every dispatcher
/// (Metal / CUDA / CPU) so effects never have to fill mip metadata manually.
pub fn make_outgoing_desc(config: &Configuration) -> TextureDesc {
	let mut desc = make_texture_desc(
		config.outgoing_width,
		config.outgoing_height,
		config.outgoing_pitch_px as u32,
		config.bytes_per_pixel,
		config.pixel_layout,
	);
	if config.outgoing_mip_levels > 1 {
		fill_mip_desc(
			&mut desc,
			config.outgoing_width,
			config.outgoing_height,
			config.outgoing_pitch_px as u32,
			config.bytes_per_pixel,
			config.outgoing_mip_levels,
		);
	}
	desc
}

/// Populate a [`TextureDesc`] with a tightly packed mip chain of `levels`
/// levels. Level 0 keeps the caller-provided pitch so it stays byte-compatible
/// with a plain (non-mip) buffer; levels 1..N use tight pitches so the total
/// byte budget equals [`mip_buffer_size_bytes`] exactly.
pub fn fill_mip_desc(desc: &mut TextureDesc, width: u32, height: u32, pitch_px: u32, bpp: u32, levels: u32) {
	let n = levels.max(1).min(MAX_MIP);
	desc.mip_level_count = n;
	desc.mip_offset_bytes = [0; MAX_MIP as usize];
	desc.mip_width = [0; MAX_MIP as usize];
	desc.mip_height = [0; MAX_MIP as usize];
	desc.mip_pitch_bytes = [0; MAX_MIP as usize];

	// Level 0 uses the host pitch; everything below is tightly packed
	// starting at the byte right after level 0 finishes.
	desc.mip_width[0] = width;
	desc.mip_height[0] = height;
	desc.mip_pitch_bytes[0] = pitch_px * bpp;
	desc.mip_offset_bytes[0] = 0;

	let mut off = pitch_px * bpp * height;
	for i in 1..n as usize {
		let lvl = i as u32;
		let w = (width >> lvl).max(1);
		let h = (height >> lvl).max(1);
		desc.mip_offset_bytes[i] = off;
		desc.mip_width[i] = w;
		desc.mip_height[i] = h;
		desc.mip_pitch_bytes[i] = w * bpp;
		off = off.saturating_add(w * h * bpp);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn default_texture_desc_has_level0_populated() {
		let d = make_texture_desc(1920, 1080, 1920, 4, 1);
		assert_eq!(d.mip_level_count, 1);
		assert_eq!(d.mip_width[0], 1920);
		assert_eq!(d.mip_height[0], 1080);
		assert_eq!(d.mip_pitch_bytes[0], 1920 * 4);
		assert_eq!(d.mip_offset_bytes[0], 0);
	}

	#[test]
	fn mip_buffer_size_matches_sum_of_levels() {
		// 32x32 @ 4 bpp, 3 levels: 32*32 + 16*16 + 8*8 = 1024 + 256 + 64 = 1344
		let size = mip_buffer_size_bytes(32, 32, 4, 3);
		assert_eq!(size, (1024 + 256 + 64) * 4);
	}

	#[test]
	fn fill_mip_desc_chains_offsets_tightly() {
		let mut d = make_texture_desc(32, 32, 32, 4, 1);
		fill_mip_desc(&mut d, 32, 32, 32, 4, 3);
		assert_eq!(d.mip_level_count, 3);
		// Level 0: 32 rows * 128 B = 4096 B
		assert_eq!(d.mip_offset_bytes[0], 0);
		assert_eq!(d.mip_pitch_bytes[0], 128);
		// Level 1: starts at 4096, 16*16*4 = 1024 B
		assert_eq!(d.mip_offset_bytes[1], 4096);
		assert_eq!(d.mip_width[1], 16);
		assert_eq!(d.mip_height[1], 16);
		assert_eq!(d.mip_pitch_bytes[1], 64);
		// Level 2: starts at 4096+1024 = 5120, 8*8*4 = 256 B
		assert_eq!(d.mip_offset_bytes[2], 5120);
		assert_eq!(d.mip_width[2], 8);
		assert_eq!(d.mip_height[2], 8);
		assert_eq!(d.mip_pitch_bytes[2], 32);
	}

	#[test]
	fn mip_buffer_size_clamps_levels() {
		// Asking for 0 levels still returns a non-zero size (min 1 level).
		let size = mip_buffer_size_bytes(32, 32, 4, 0);
		assert_eq!(size, 32 * 32 * 4);
		// Asking past MAX_MIP is clamped to MAX_MIP.
		let size_large = mip_buffer_size_bytes(64, 64, 4, 999);
		let expected: u32 = (0..MAX_MIP).map(|l| ((64u32 >> l).max(1)) * ((64u32 >> l).max(1)) * 4).sum();
		assert_eq!(size_large, expected);
	}

	#[test]
	fn rust_texture_desc_size_matches_slang_layout() {
		// 7 scalar u32 + 1 level count + 4 * [u32; 5] = (7 + 1 + 20) * 4 = 112 bytes.
		assert_eq!(std::mem::size_of::<TextureDesc>(), (7 + 1 + 20) * 4);
	}
}
