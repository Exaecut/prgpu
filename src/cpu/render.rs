use std::ffi::c_void;

use after_effects::{self as ae, log};

use crate::types::{Configuration, FrameParams};

/// Per-pixel VEKL dispatch function type.
///
/// Signature matches the generated C++ per-pixel entry point:
/// ```cpp
/// void name_cpu_dispatch(
///     unsigned int gid_x, unsigned int gid_y,
///     const void* const* buffers,
///     const void* transition_params,
///     const void* user_params
/// );
/// ```
pub type CpuDispatchFn = unsafe extern "C" fn(
	u32,                  // gid_x
	u32,                  // gid_y
	*const *const c_void, // buffers
	*const c_void,        // transition_params (FrameParams*)
	*const c_void,        // user_params
);

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
/// For After Effects, always returns 1 (BGRA).
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

/// Unified CPU render dispatch.
///
/// Iterates over every pixel, calling `dispatch_fn` for each one.
/// - **After Effects** with matching Layer dimensions: uses `iterate_with`
///   (AE multi-threaded iterate suites, supports 8/16/32-bit via
///   Iterate8/Iterate16/IterateFloat)
/// - **Premiere** or mismatched dimensions (blur intermediate buffers):
///   uses rayon parallel row iteration proven pattern:
///   `out_buf.par_chunks_mut(stride).enumerate().for_each(|(y, row)| { row.chunks_mut(bpp)... })`
pub fn render_cpu<P: Copy + Sync>(
	in_data: &ae::InData,
	in_layer: &ae::Layer,
	out_layer: &mut ae::Layer,
	config: &Configuration,
	dispatch_fn: CpuDispatchFn,
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

	log::info!(
		"render_cpu: {}x{} bpp={} out_pitch={}px in_pitch={}px dest_pitch={}px outgoing={:?} incoming={:?} dest={:?}",
		w,
		h,
		tp.bpp,
		tp.out_pitch,
		tp.in_pitch,
		tp.dest_pitch,
		outgoing_ptr,
		incoming_ptr,
		dest_ptr
	);

	// Use iterate_with when: AE host AND config dimensions match the output Layer.
	// This ensures the AE iterate suites iterate over the correct pixel range.
	// For blur intermediate buffers (dimensions differ), fall through to rayon.
	let can_iterate_with = !in_data.is_premiere() && w == out_layer.width() as u32 && h == out_layer.height() as u32;

	if can_iterate_with {
		log::info!(
			"dispatch path: ae (iterate_with), layer={}x{} config={}x{}",
			out_layer.width(),
			out_layer.height(),
			w,
			h
		);
		ae_dispatch(in_layer, out_layer, buffers, tp, user_params, dispatch_fn)
	} else {
		log::info!(
			"dispatch path: rayon, is_premiere={}, config={}x{} layer={}x{}",
			in_data.is_premiere(),
			w,
			h,
			out_layer.width(),
			out_layer.height()
		);

		// sdk_noise pattern: get buffer slices from Layers and keep them alive
		// for the entire dispatch. This ensures the host buffer remains
		// mapped/locked in Premiere for the duration of the parallel iteration.
		// Get strides BEFORE mutable borrow to satisfy borrow checker.
		let _in_stride_bytes = in_layer.buffer_stride();
		let out_stride_bytes = out_layer.buffer_stride();
		let in_buf = in_layer.buffer();
		let mut out_buf = out_layer.buffer_mut();

		// Use fresh pointers from kept-alive slices for layer-backed buffers.
		// In Premiere, buffer_mut() may map/lock the host buffer, so the pointer
		// from a fresh call may differ from the stale config pointer captured earlier.
		let fresh_outgoing = in_buf.as_ptr() as *const c_void;
		let fresh_dest = out_buf.as_ptr() as *const c_void;
		// For incoming: if config uses the same pointer as outgoing (no-blur case),
		// use the fresh outgoing pointer; otherwise keep config's intermediate buffer pointer.
		let fresh_incoming = if incoming_ptr == outgoing_ptr {
			fresh_outgoing
		} else {
			incoming_ptr
		};
		let fresh_buffers = SafeBuffers([fresh_outgoing, fresh_incoming, fresh_dest]);

		// Pointer verification: config buffer pointers should match actual Layer buffer pointers.
		// A mismatch indicates stale pointers from dropped temporary slices.
		log::info!(
			"rayon ptr check: in_buf={:?} vs config.outgoing={:?} match={} | out_buf={:?} vs config.dest={:?} match={}",
			fresh_outgoing, outgoing_ptr, fresh_outgoing == outgoing_ptr,
			fresh_dest, dest_ptr, fresh_dest == dest_ptr
		);

		rayon_dispatch(w, h, fresh_buffers, tp, user_params, dispatch_fn, &mut out_buf, &in_buf, out_stride_bytes)
	}
}

/// AE path: `iterate_with` for multi-threaded dispatch.
///
/// Supports 8-bit (Iterate8Suite), 16-bit (Iterate16Suite), and
/// 32-bit float (IterateFloatSuite) automatically via `GenericPixel` dispatch.
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
				log::info!("ae_dispatch: first callback fired at ({}, {})", x, y);
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

/// Rayon path: parallel row iteration following sdk_noise's proven pattern.
///
/// Uses `par_chunks_mut(out_stride)` on the output buffer to keep it alive/mapped
/// for the entire dispatch, matching how sdk_noise iterates pixels in Premiere.
/// The VEKL dispatch function writes through `buffers[2]` (dest), which points
/// to the same memory as `out_buf`.
///
/// Used for Premiere (no IterateFloat support) and for intermediate buffer
/// dispatch where config dimensions don't match Layer dimensions
/// (e.g., blur downsampled buffers).
fn rayon_dispatch<P: Copy + Sync>(
	width: u32,
	height: u32,
	buffers: SafeBuffers,
	tp: FrameParams,
	user_params: &P,
	dispatch_fn: CpuDispatchFn,
	out_buf: &mut [u8],
	_in_buf: &[u8],
	out_stride_bytes: usize,
) -> Result<(), ae::Error> {
	use rayon::prelude::*;
	use std::sync::atomic::{AtomicBool, Ordering};

	let first_call = AtomicBool::new(true);
	// Convert pointers to usize so the closure captures plain usize values
	// (raw pointers are not Send/Sync, but usize is).
	let buf_ptr = buffers.0.as_ptr() as usize;
	let tp_ptr = &tp as *const _ as usize;
	let up_ptr = user_params as *const _ as usize;

	log::info!("rayon_dispatch: {}x{} out_stride={}b bpp={}", width, height, out_stride_bytes, tp.bpp);

	// sdk_noise pattern: par_chunks_mut on output buffer keeps it alive/mapped
	// for the entire parallel iteration, ensuring Premiere's host buffer stays valid.
	out_buf.par_chunks_mut(out_stride_bytes).enumerate().for_each(move |(y, _row_bytes)| {
		if first_call.swap(false, Ordering::Relaxed) {
			log::info!("rayon_dispatch: first callback fired at row y={}", y);
		}
		for x in 0..width {
			unsafe {
				dispatch_fn(x as u32, y as u32, buf_ptr as *const *const c_void, tp_ptr as *const c_void, up_ptr as *const c_void);
			}
		}
	});

	Ok(())
}
