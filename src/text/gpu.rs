//! Atlas upload + bounded-dispatch host side of text rendering.
//!
//! The SDF atlas is built once on the CPU and uploaded once per GPU device into
//! persistent device buffers (kept out of the LRU image pool so they're never
//! evicted). [`draw`] lays the string out on the CPU, computes its pixel
//! bounding box, and dispatches the `text_overlay` kernel over that box only.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::kernel::builtin::{text_overlay, TextOverlayParams};
use crate::text::atlas::{build_default_atlas, Atlas, FIRST_CHAR, GLYPH_COUNT, LAST_CHAR};
use crate::types::Configuration;

/// A single line of text to composite onto the destination frame.
#[derive(Clone, Debug)]
pub struct TextSpec {
	/// Text box top-left in destination pixels.
	pub x: f32,
	pub y: f32,
	/// Cap height of the text in destination pixels.
	pub px_size: f32,
	/// Straight (non-premultiplied) RGBA text colour, 0..1.
	pub color: [f32; 4],
	pub text: String,
}

/// Persistent per-device GPU buffers for the font. Raw device handles
/// (CUDA `CUdeviceptr` or Metal `MTLBuffer`), kept alive for the process.
struct GpuFont {
	atlas: *mut c_void,
	metrics: *mut c_void,
}
unsafe impl Send for GpuFont {}
unsafe impl Sync for GpuFont {}

fn cpu_atlas() -> &'static Atlas {
	static CPU_ATLAS: OnceLock<Atlas> = OnceLock::new();
	CPU_ATLAS.get_or_init(build_default_atlas)
}

fn uploads() -> &'static Mutex<HashMap<usize, GpuFont>> {
	static UPLOADS: OnceLock<Mutex<HashMap<usize, GpuFont>>> = OnceLock::new();
	UPLOADS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Device cache key: the CUDA context (CUDA) or the MTLDevice (Metal).
fn device_key(config: &Configuration) -> usize {
	#[cfg(gpu_backend = "cuda")]
	{
		config.context_handle.map(|p| p as usize).unwrap_or(0)
	}
	#[cfg(not(gpu_backend = "cuda"))]
	{
		config.device_handle as usize
	}
}

/// Composite `spec`'s text onto the destination frame described by `config`.
/// GPU-only; a CPU-backend `config` is a no-op (overlay is a GPU feature).
pub fn draw(config: &Configuration, spec: &TextSpec) -> Result<(), &'static str> {
	if config.context_handle.is_none() || config.dest_data.is_null() {
		return Ok(());
	}

	let atlas = cpu_atlas();
	let font = ensure_uploaded(config, atlas)?;

	let Some(params) = layout(atlas, spec, config.width, config.height) else {
		return Ok(());
	};
	if params.bbox_w == 0 || params.bbox_h == 0 {
		return Ok(());
	}

	// Bounded variant of the frame config: atlas/metrics in the source slots,
	// destination + pitch/storage/layout left pointing at the real frame, and
	// width/height set to the bbox so the dispatch grid covers only the text.
	let mut cfg = *config;
	cfg.outgoing_data = Some(font.atlas);
	cfg.incoming_data = Some(font.metrics);
	cfg.outgoing_width = atlas.width;
	cfg.outgoing_height = atlas.height;
	cfg.outgoing_pitch_px = atlas.width as i32;
	cfg.incoming_width = atlas.width;
	cfg.incoming_height = atlas.height;
	cfg.width = params.bbox_w;
	cfg.height = params.bbox_h;
	cfg.outgoing_mip_levels = 0;

	unsafe { text_overlay::kernel().dispatch_gpu(&cfg, params) }
}

/// Lay the single line out left-to-right and build the kernel params + bbox.
/// `(x, y)` is the text box top-left; the baseline sits `ascent*scale` below.
fn layout(atlas: &Atlas, spec: &TextSpec, frame_w: u32, frame_h: u32) -> Option<TextOverlayParams> {
	let scale = (spec.px_size / atlas.base_px).max(1e-4);
	let pen_x0 = spec.x;
	let pen_y = spec.y + atlas.ascent * scale;

	let mut pen_x = pen_x0;
	let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
	let mut packed = [0u32; 64];
	let mut count = 0usize;

	for ch in spec.text.chars().take(256) {
		let code = ch as u32;
		let in_range = (FIRST_CHAR..=LAST_CHAR).contains(&code);
		let gi = if in_range { (code - FIRST_CHAR) as usize } else { 0 };
		let stored = if in_range { code } else { FIRST_CHAR };
		packed[count >> 2] |= (stored & 0xFF) << ((count & 3) * 8);

		let m = atlas.metrics[gi];
		if m.cell_w > 0.0 && m.cell_h > 0.0 {
			let cx0 = pen_x + m.left * scale;
			let cy0 = pen_y + m.top * scale;
			min_x = min_x.min(cx0);
			min_y = min_y.min(cy0);
			max_x = max_x.max(cx0 + m.cell_w * scale);
			max_y = max_y.max(cy0 + m.cell_h * scale);
		}
		pen_x += m.advance * scale;
		count += 1;
	}

	if count == 0 || max_x < min_x {
		return None;
	}

	let bx0 = (min_x.floor().max(0.0)) as i64;
	let by0 = (min_y.floor().max(0.0)) as i64;
	let bx1 = (max_x.ceil() as i64).min(frame_w as i64);
	let by1 = (max_y.ceil() as i64).min(frame_h as i64);
	let bbox_w = (bx1 - bx0).max(0) as u32;
	let bbox_h = (by1 - by0).max(0) as u32;

	Some(TextOverlayParams {
		color: spec.color,
		pen_x: pen_x0,
		pen_y,
		scale,
		spread: atlas.spread,
		atlas_w: atlas.width,
		atlas_h: atlas.height,
		frame_w,
		frame_h,
		bbox_x: bx0 as u32,
		bbox_y: by0 as u32,
		bbox_w,
		bbox_h,
		char_count: count as u32,
		first_char: FIRST_CHAR,
		glyph_count: GLYPH_COUNT as u32,
		_pad0: 0,
		packed,
	})
}

fn ensure_uploaded(config: &Configuration, atlas: &Atlas) -> Result<GpuFont, &'static str> {
	let key = device_key(config);
	let mut guard = uploads().lock();
	if let Some(f) = guard.get(&key) {
		return Ok(GpuFont { atlas: f.atlas, metrics: f.metrics });
	}

	let metrics_bytes: &[u8] = unsafe { std::slice::from_raw_parts(atlas.metrics.as_ptr() as *const u8, std::mem::size_of_val(atlas.metrics.as_slice())) };

	let font = unsafe { upload_font(config, &atlas.pixels, metrics_bytes)? };
	guard.insert(key, GpuFont { atlas: font.atlas, metrics: font.metrics });
	Ok(font)
}

/// # Safety: `config` device/context handles must be valid for the active backend.
#[cfg(gpu_backend = "cuda")]
unsafe fn upload_font(config: &Configuration, atlas_bytes: &[u8], metrics_bytes: &[u8]) -> Result<GpuFont, &'static str> {
	use cudarc::driver::sys::{self as cuda, cuMemAlloc_v2, cuMemcpyHtoD_v2, CUdeviceptr, CUresult};

	let ctx = config.context_handle.ok_or("text: no CUDA context")?;
	unsafe { cuda::cuCtxSetCurrent(ctx as cuda::CUcontext) };

	let alloc = |bytes: &[u8]| -> Result<*mut c_void, &'static str> {
		let mut dptr: CUdeviceptr = 0;
		if unsafe { cuMemAlloc_v2(&mut dptr, bytes.len()) } != CUresult::CUDA_SUCCESS {
			return Err("text: cuMemAlloc_v2 failed");
		}
		if unsafe { cuMemcpyHtoD_v2(dptr, bytes.as_ptr() as *const c_void, bytes.len()) } != CUresult::CUDA_SUCCESS {
			return Err("text: cuMemcpyHtoD_v2 failed");
		}
		Ok(dptr as *mut c_void)
	};

	Ok(GpuFont { atlas: alloc(atlas_bytes)?, metrics: alloc(metrics_bytes)? })
}

/// # Safety: `config.device_handle` must be a valid MTLDevice.
#[cfg(gpu_backend = "metal")]
unsafe fn upload_font(config: &Configuration, atlas_bytes: &[u8], metrics_bytes: &[u8]) -> Result<GpuFont, &'static str> {
	use objc::{msg_send, runtime::Object, sel, sel_impl};

	let device = config.device_handle as *mut Object;
	if device.is_null() {
		return Err("text: null MTLDevice");
	}
	// MTLResourceStorageModeShared = 0; buffer is retained and kept forever.
	let make = |bytes: &[u8]| -> Result<*mut c_void, &'static str> {
		let buf: *mut Object = unsafe { msg_send![device, newBufferWithBytes: bytes.as_ptr() as *const c_void length: bytes.len() options: 0u64] };
		if buf.is_null() { Err("text: newBufferWithBytes failed") } else { Ok(buf as *mut c_void) }
	};
	Ok(GpuFont { atlas: make(atlas_bytes)?, metrics: make(metrics_bytes)? })
}

#[cfg(not(any(gpu_backend = "cuda", gpu_backend = "metal")))]
unsafe fn upload_font(_config: &Configuration, _atlas_bytes: &[u8], _metrics_bytes: &[u8]) -> Result<GpuFont, &'static str> {
	Err("text: no GPU backend")
}
