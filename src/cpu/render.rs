use std::ffi::c_void;

use after_effects as ae;

use crate::types::{Configuration, FrameParams};

pub type CpuDispatchFn = unsafe extern "C" fn(u32, u32, *const *const c_void, *const c_void, *const c_void);

/// Row-batch dispatch: processes an entire row in a single C call.
/// Sets row-invariant TLS once and loops x internally, eliminating
/// (width-1) × 4 redundant TLS writes + (width-1) FFI calls per row.
pub type CpuRowBatchFn = unsafe extern "C" fn(u32, u32, *const *const c_void, *const c_void, *const c_void);

/// Holds both dispatch function pointers for a kernel.
/// `per_pixel` is the original per-pixel dispatch (used by ae_dispatch).
/// `row_batch` is the per-row batch dispatch (used by rayon_dispatch).
#[derive(Copy, Clone, Debug)]
pub struct CpuDispatchFns {
	pub per_pixel: CpuDispatchFn,
	pub row_batch: CpuRowBatchFn,
}

/// Wrapper to make buffer pointer array `Send + Sync`.
///
/// SAFETY: The buffer pointers are valid for the duration of the dispatch call.
/// They point to Layer-backed or intermediate buffers that outlive the iteration.
#[derive(Copy, Clone, Debug)]
struct SafeBuffers([*const c_void; 3]);
unsafe impl Send for SafeBuffers {}
unsafe impl Sync for SafeBuffers {}

/// Maps a Premiere `PixelFormat` to the VEKL layout type code.
///
/// - 0 = RGBA (identity)
/// - 1 = BGRA (channel swap B↔R)
/// - 2 = VUYA BT.601 (YCbCr→RGB)
/// - 3 = VUYA BT.709 (YCbCr→RGB)
///
/// For After Effects, always returns 1 (BGRA
pub fn pixel_layout_from_format(in_data: &ae::InData, layer: &ae::Layer) -> u32 {
	if in_data.is_premiere() {
		if let Ok(fmt) = layer.pr_pixel_format() {
			match fmt {
				// VUYA BT.709 variants
				ae::pr::PixelFormat::Vuya4444_8u709
				| ae::pr::PixelFormat::Vuya4444_32f709
				| ae::pr::PixelFormat::Vuyx4444_8u709
				| ae::pr::PixelFormat::Vuyx4444_32f709
				| ae::pr::PixelFormat::Vuyp4444_8u709
				| ae::pr::PixelFormat::Vuyp4444_32f709 => 3,

				// VUYA BT.601 variants
				ae::pr::PixelFormat::Vuya4444_8u
				| ae::pr::PixelFormat::Vuya4444_16u
				| ae::pr::PixelFormat::Vuya4444_32f
				| ae::pr::PixelFormat::Vuyx4444_8u
				| ae::pr::PixelFormat::Vuyx4444_32f
				| ae::pr::PixelFormat::Vuyp4444_8u
				| ae::pr::PixelFormat::Vuyp4444_32f => 2,

				// BGRA / ARGB / other RGB variants → BGRA layout
				_ => 1,
			}
		} else {
			1 // Default to BGRA for Premiere
		}
	} else {
		1 // After Effects always uses BGRA
	}
}

/// Auto-compute bytes-per-pixel from the Layer based on host.
///
/// - **After Effects**: uses `world_type()` → U8=4, U15=8, F32=16
/// - **Premiere**: uses `pr_pixel_format()` to match BGRA/VUYA × 8u/16u/32f
pub fn compute_bpp(in_data: &ae::InData, layer: &ae::Layer) -> Result<u32, ae::Error> {
	if in_data.is_premiere() {
		let fmt = layer.pr_pixel_format()?;
		match fmt {
			// 8-bit: 4 bytes per pixel (4 channels × 1 byte)
			ae::pr::PixelFormat::Bgra4444_8u
			| ae::pr::PixelFormat::Vuya4444_8u
			| ae::pr::PixelFormat::Vuya4444_8u709
			| ae::pr::PixelFormat::Argb4444_8u
			| ae::pr::PixelFormat::Bgrx4444_8u
			| ae::pr::PixelFormat::Vuyx4444_8u
			| ae::pr::PixelFormat::Vuyx4444_8u709
			| ae::pr::PixelFormat::Xrgb4444_8u
			| ae::pr::PixelFormat::Bgrp4444_8u
			| ae::pr::PixelFormat::Vuyp4444_8u
			| ae::pr::PixelFormat::Vuyp4444_8u709
			| ae::pr::PixelFormat::Prgb4444_8u => Ok(4),

			// 16-bit: 8 bytes per pixel (4 channels × 2 bytes)
			ae::pr::PixelFormat::Bgra4444_16u
			| ae::pr::PixelFormat::Vuya4444_16u
			| ae::pr::PixelFormat::Argb4444_16u
			| ae::pr::PixelFormat::Bgrx4444_16u
			| ae::pr::PixelFormat::Xrgb4444_16u
			| ae::pr::PixelFormat::Bgrp4444_16u
			| ae::pr::PixelFormat::Prgb4444_16u => Ok(8),

			// 32-bit float: 16 bytes per pixel (4 channels × 4 bytes)
			ae::pr::PixelFormat::Bgra4444_32f
			| ae::pr::PixelFormat::Vuya4444_32f
			| ae::pr::PixelFormat::Vuya4444_32f709
			| ae::pr::PixelFormat::Argb4444_32f
			| ae::pr::PixelFormat::Bgrx4444_32f
			| ae::pr::PixelFormat::Vuyx4444_32f
			| ae::pr::PixelFormat::Vuyx4444_32f709
			| ae::pr::PixelFormat::Xrgb4444_32f
			| ae::pr::PixelFormat::Bgrp4444_32f
			| ae::pr::PixelFormat::Vuyp4444_32f
			| ae::pr::PixelFormat::Vuyp4444_32f709
			| ae::pr::PixelFormat::Prgb4444_32f
			| ae::pr::PixelFormat::Bgra4444_32fLinear
			| ae::pr::PixelFormat::Bgrp4444_32fLinear
			| ae::pr::PixelFormat::Bgrx4444_32fLinear
			| ae::pr::PixelFormat::Argb4444_32fLinear
			| ae::pr::PixelFormat::Prgb4444_32fLinear
			| ae::pr::PixelFormat::Xrgb4444_32fLinear => Ok(16),

			_ => Err(ae::Error::InvalidParms),
		}
	} else {
		match layer.world_type() {
			ae::aegp::WorldType::U8 => Ok(4),
			ae::aegp::WorldType::U15 => Ok(8),
			ae::aegp::WorldType::F32 => Ok(16),
			_ => Err(ae::Error::Generic),
		}
	}
}

pub fn render_cpu<P: Copy + Sync>(
	kernel_name: &'static str,
	in_data: &ae::InData,
	in_layer: &ae::Layer,
	out_layer: &mut ae::Layer,
	config: &Configuration,
	dispatch_fns: CpuDispatchFns,
	user_params: &P,
) -> Result<(), ae::Error> {
	let w = config.width;
	let h = config.height;
	if w == 0 || h == 0 {
		return Ok(());
	}

	let outgoing_ptr = config.outgoing_data.unwrap_or(std::ptr::null_mut()) as *const c_void;
	let incoming_ptr = config.incoming_data.unwrap_or(std::ptr::null_mut()) as *const c_void;
	let dest_ptr = config.dest_data as *const c_void;

	let buffers = SafeBuffers([outgoing_ptr, incoming_ptr, dest_ptr]);

	let tp = FrameParams {
		out_pitch: config.outgoing_pitch_px as u32,
		in_pitch: config.incoming_pitch_px as u32,
		dest_pitch: config.dest_pitch_px as u32,
		width: w,
		height: h,
		progress: config.progress,
		bpp: config.bytes_per_pixel,
		pixel_layout: config.pixel_layout,
	};

	let can_iterate_with = !in_data.is_premiere() && w == out_layer.width() as u32 && h == out_layer.height() as u32;

	let start = std::time::Instant::now();

	let result = if can_iterate_with {
		ae_dispatch(in_layer, out_layer, buffers, tp, user_params, dispatch_fns.per_pixel)
	} else {
		// Use config-provided buffer pointers directly.
		// Do NOT replace with AE layer pointers — intermediate buffers
		// (e.g., blur temporaries) have their own pointers that must be respected.
		let out_stride_bytes = (tp.dest_pitch * tp.bpp) as usize;
		let out_buf_size = (h as usize) * out_stride_bytes;

		// SAFETY: dest_ptr points to a buffer of at least out_buf_size bytes,
		// as guaranteed by the caller via Configuration. The slice is only used
		// to partition row iteration across rayon threads; actual pixel I/O
		// goes through the dispatch function's buffer pointer array.
		let out_buf = if out_buf_size > 0 && !dest_ptr.is_null() {
			unsafe { std::slice::from_raw_parts_mut(dest_ptr as *mut u8, out_buf_size) }
		} else {
			&mut []
		};
		let in_buf = in_layer.buffer();

		rayon_dispatch(w, h, buffers, tp, user_params, dispatch_fns.row_batch, out_buf, in_buf, out_stride_bytes)
	};

	crate::timing::record(kernel_name, crate::timing::Backend::Cpu, start.elapsed().as_nanos() as u64);

	result
}

fn ae_dispatch<P: Copy + Sync>(
	in_layer: &ae::Layer,
	out_layer: &mut ae::Layer,
	buffers: SafeBuffers,
	tp: FrameParams,
	user_params: &P,
	dispatch_fn: CpuDispatchFn,
) -> Result<(), ae::Error> {
	let first_call = std::cell::Cell::new(true);
	in_layer.iterate_with(
		out_layer,
		0,
		tp.height as i32,
		None,
		move |x: i32, y: i32, _pixel: ae::GenericPixel, _out_pixel: ae::GenericPixelMut| {
			if first_call.get() {
				first_call.set(false);
			}

			unsafe {
				dispatch_fn(
					x as u32,
					y as u32,
					buffers.0.as_ptr(),
					&tp as *const _ as *const c_void,
					user_params as *const _ as *const c_void,
				);
			}
			Ok(())
		},
	)
}

/// Row-batch rayon dispatch: one FFI call per row instead of per pixel.
///
/// The C-side `_cpu_row_dispatch` function sets row-invariant TLS variables
/// (`__cpu_dispatch_w`, `__cpu_dispatch_h`, `__cpu_format`, `__cpu_gid_y`)
/// once and loops over x internally. This eliminates:
/// - (width-1) × 4 redundant TLS writes per row
/// - (width-1) Rust→C cross-language calls per row
fn rayon_dispatch<P: Copy + Sync>(
	width: u32,
	_height: u32,
	buffers: SafeBuffers,
	tp: FrameParams,
	user_params: &P,
	row_batch_fn: CpuRowBatchFn,
	out_buf: &mut [u8],
	_in_buf: &[u8],
	out_stride_bytes: usize,
) -> Result<(), ae::Error> {
	use rayon::prelude::*;

	let buf_ptr = buffers.0.as_ptr() as usize;
	let tp_ptr = &tp as *const _ as usize;
	let up_ptr = user_params as *const _ as usize;

	out_buf.par_chunks_mut(out_stride_bytes).enumerate().for_each(move |(y, _row_bytes)| {
		unsafe {
			row_batch_fn(
				y as u32,
				width,
				buf_ptr as *const *const c_void,
				tp_ptr as *const c_void,
				up_ptr as *const c_void,
			);
		}
	});

	Ok(())
}
