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
	/// Cap height of the text in destination pixels.
	pub px_size: f32,
	/// Straight (non-premultiplied) RGBA text colour, 0..1.
	pub color: [f32; 4],
	/// Placement of the text (and optional background band).
	pub layout: TextLayout,
	pub text: String,
}

/// How [`TextSpec`] is placed in the destination frame.
#[derive(Clone, Copy, Debug)]
pub enum TextLayout {
	/// Free placement: text box top-left at `(x, y)` in destination pixels,
	/// no background band.
	At { x: f32, y: f32 },
	/// A background band sized as a percentage (0..100) of the output canvas
	/// and centered in it; the text is centered within the band. `background`
	/// is straight RGBA (alpha 0 = no visible band).
	Banner { width_pct: f32, height_pct: f32, background: [f32; 4] },
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

/// Lay the single line out left-to-right and build the kernel params + dispatch
/// box. The dispatch box is the text bbox (`At`) or the centered band
/// (`Banner`); the kernel paints the band background, then the glyphs.
fn layout(atlas: &Atlas, spec: &TextSpec, frame_w: u32, frame_h: u32) -> Option<TextOverlayParams> {
	let scale = (spec.px_size / atlas.base_px).max(1e-4);
	let fw = frame_w as f32;
	let fh = frame_h as f32;

	// Pass 1: pack char codes and measure the advance width.
	let mut packed = [0u32; 64];
	let mut count = 0usize;
	let mut text_width = 0.0f32;
	for ch in spec.text.chars().take(256) {
		let code = ch as u32;
		let in_range = (FIRST_CHAR..=LAST_CHAR).contains(&code);
		let gi = if in_range { (code - FIRST_CHAR) as usize } else { 0 };
		let stored = if in_range { code } else { FIRST_CHAR };
		packed[count >> 2] |= (stored & 0xFF) << ((count & 3) * 8);
		text_width += atlas.metrics[gi].advance * scale;
		count += 1;
	}
	if count == 0 {
		return None;
	}

	// Visual height from baseline metrics (descent is negative).
	let text_h = (atlas.ascent - atlas.descent) * scale;
	let margin = (atlas.spread * scale).ceil();

	let (bbox_x, bbox_y, bbox_w, bbox_h, pen_x0, baseline_y, bg_color) = match spec.layout {
		TextLayout::At { x, y } => {
			let baseline = y + atlas.ascent * scale;
			let bx0 = (x - margin).floor().max(0.0);
			let by0 = (y - margin).floor().max(0.0);
			let bx1 = (x + text_width + margin).ceil().min(fw);
			let by1 = (y + text_h + margin).ceil().min(fh);
			(bx0 as u32, by0 as u32, (bx1 - bx0).max(0.0) as u32, (by1 - by0).max(0.0) as u32, x, baseline, [0.0; 4])
		}
		TextLayout::Banner { width_pct, height_pct, background } => {
			let band_w = (width_pct / 100.0).clamp(0.0, 1.0) * fw;
			let band_h = (height_pct / 100.0).clamp(0.0, 1.0) * fh;
			let band_x = ((fw - band_w) / 2.0).max(0.0);
			let band_y = ((fh - band_h) / 2.0).max(0.0);
			// Center the text in the band on both axes.
			let pen_x = band_x + (band_w - text_width) / 2.0;
			let baseline = band_y + (band_h - text_h) / 2.0 + atlas.ascent * scale;
			(band_x.floor() as u32, band_y.floor() as u32, band_w.ceil().min(fw) as u32, band_h.ceil().min(fh) as u32, pen_x, baseline, background)
		}
	};

	if bbox_w == 0 || bbox_h == 0 {
		return None;
	}

	Some(TextOverlayParams {
		color: spec.color,
		bg_color,
		pen_x: pen_x0,
		pen_y: baseline_y,
		scale,
		spread: atlas.spread,
		atlas_w: atlas.width,
		atlas_h: atlas.height,
		frame_w,
		frame_h,
		bbox_x,
		bbox_y,
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
