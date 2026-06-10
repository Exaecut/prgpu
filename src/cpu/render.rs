use std::ffi::c_void;

use after_effects as ae;

use crate::types::{Configuration, FrameParams};

/// Per-pixel CPU dispatch. Used by the AE `iterate_with` path which drives `(x, y)` externally.
pub type CpuDispatchFn = unsafe extern "C" fn(u32, u32, *const *const c_void, *const c_void, *const c_void);

/// Tile CPU dispatch. One FFI call per rayon chunk amortizes the boundary across `rows_per_task × width` invocations.
pub type CpuDispatchTileFn = unsafe extern "C" fn(u32, u32, u32, *const *const c_void, *const c_void, *const c_void);

/// `Send + Sync` wrapper for the buffer pointer array.
///
/// SAFETY: pointers are valid for the dispatch and outlive the iteration.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SafeBuffers(pub(crate) [*const c_void; 3]);
unsafe impl Send for SafeBuffers {}
unsafe impl Sync for SafeBuffers {}

/// Map a Premiere `PixelFormat` to the VEKL layout id.
///
/// 0 = RGBA, 1 = BGRA, 2 = VUYA BT.601, 3 = VUYA BT.709. After Effects always returns 1 (BGRA).
pub fn pixel_layout_from_format(in_data: &ae::InData, layer: &ae::Layer) -> u32 {
	if in_data.is_premiere() {
		if let Ok(fmt) = layer.pr_pixel_format() {
			match fmt {
				ae::pr::PixelFormat::Vuya4444_8u709
				| ae::pr::PixelFormat::Vuya4444_32f709
				| ae::pr::PixelFormat::Vuyx4444_8u709
				| ae::pr::PixelFormat::Vuyx4444_32f709
				| ae::pr::PixelFormat::Vuyp4444_8u709
				| ae::pr::PixelFormat::Vuyp4444_32f709 => 3,

				ae::pr::PixelFormat::Vuya4444_8u
				| ae::pr::PixelFormat::Vuya4444_16u
				| ae::pr::PixelFormat::Vuya4444_32f
				| ae::pr::PixelFormat::Vuyx4444_8u
				| ae::pr::PixelFormat::Vuyx4444_32f
				| ae::pr::PixelFormat::Vuyp4444_8u
				| ae::pr::PixelFormat::Vuyp4444_32f => 2,

				_ => 1,
			}
		} else {
			1 // Premiere default: BGRA
		}
	} else {
		1 // AE: always BGRA
	}
}

/// Bytes per pixel from the layer's pixel format.
/// AE: `world_type()` (U8=4, U15=8, F32=16). Premiere: `pr_pixel_format()`.
pub fn compute_bpp(in_data: &ae::InData, layer: &ae::Layer) -> Result<u32, ae::Error> {
	if in_data.is_premiere() {
		let fmt = layer.pr_pixel_format()?;
		match fmt {
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

			ae::pr::PixelFormat::Bgra4444_16u
			| ae::pr::PixelFormat::Vuya4444_16u
			| ae::pr::PixelFormat::Argb4444_16u
			| ae::pr::PixelFormat::Bgrx4444_16u
			| ae::pr::PixelFormat::Xrgb4444_16u
			| ae::pr::PixelFormat::Bgrp4444_16u
			| ae::pr::PixelFormat::Prgb4444_16u => Ok(8),

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
	dispatch_fn: CpuDispatchFn,
	dispatch_tile_fn: CpuDispatchTileFn,
	user_params: &P,
) -> Result<(), ae::Error> {
	use crate::cpu::diag;

	let w = config.width;
	let h = config.height;
	if w == 0 || h == 0 {
		return Ok(());
	}

	// Wall clock starts here; `setup_ns` covers everything before the rayon / AE body.
	let guard = diag::DispatchGuard::enter();
	let wall_start = std::time::Instant::now();

	let outgoing_ptr = config.outgoing_data.unwrap_or(std::ptr::null_mut()) as *const c_void;
	let incoming_ptr = config.incoming_data.unwrap_or(std::ptr::null_mut()) as *const c_void;
	let dest_ptr = config.dest_data as *const c_void;

	let buffers = SafeBuffers([outgoing_ptr, incoming_ptr, dest_ptr]);

	let time = if in_data.time_scale() != 0 {
		in_data.current_time() as f32 / in_data.time_scale() as f32
	} else {
		0.0
	};

	// out_desc/in_desc describe SOURCE buffers (may be downsampled); dst_desc + width/height drive the destination iteration extent.
	let mut tp = FrameParams::from_config(config);
	tp.time = time;

	let can_iterate_with = !in_data.is_premiere() && w == out_layer.width() as u32 && h == out_layer.height() as u32;

	let setup_ns = wall_start.elapsed().as_nanos() as u64;
	let body_start = std::time::Instant::now();

	let (path, chunk_rows, result) = if can_iterate_with {
		// AE `iterate_with` drives (x, y) externally; use the per-pixel entry.
		(
			diag::DispatchPath::AeIterate,
			1u32,
			ae_dispatch(in_layer, out_layer, buffers, tp, user_params, dispatch_fn),
		)
	} else {
		let out_stride_bytes = tp.dst_desc.pitch_bytes as usize;
		let out_buf_size = (h as usize) * out_stride_bytes;

		// SAFETY: caller's `Configuration` guarantees `dest_ptr` covers `out_buf_size` bytes; the slice is only used to partition rows across rayon workers.
		let out_buf = if out_buf_size > 0 && !dest_ptr.is_null() {
			unsafe { std::slice::from_raw_parts_mut(dest_ptr as *mut u8, out_buf_size) }
		} else {
			&mut []
		};

		let rows = rayon_dispatch_tile(w, buffers, tp, user_params, dispatch_tile_fn, out_buf, out_stride_bytes);
		(diag::DispatchPath::Rayon, rows, Ok(()))
	};

	let body_ns = body_start.elapsed().as_nanos() as u64;
	crate::timing::record(kernel_name, crate::types::Backend::Cpu, setup_ns + body_ns);
	diag::log_dispatch(kernel_name, path, w, h, chunk_rows, setup_ns, body_ns, guard.concurrent_at_entry());
	drop(guard);

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


/// Rows per rayon task.
///
/// Targets ~4 tasks per worker thread — coarse enough to amortize fork-join overhead
/// over the per-pixel inner loop, fine enough for good load balancing.
#[inline]
fn compute_rows_per_task(height: u32) -> u32 {
	// Chunk against the bounded render pool, not the global rayon pool, so granularity matches the pool we actually dispatch on.
	let threads = crate::cpu::pool::worker_count().max(1) as u32;
	let target_tasks = threads.saturating_mul(4).max(1);
	((height + target_tasks - 1) / target_tasks).max(1)
}

/// AE-free rayon tile dispatcher. Shared by Premiere render and the bench harness.
///
/// Calls `dispatch_tile_fn` once per rayon chunk; the C side loops over `[y0, y1) × [0, width)`.
/// Eliminates the per-pixel FFI boundary that, on Windows DLLs with dynamic-TLS,
/// was costing ~100 ns/pixel (~350 ms per 3.57 Mpx frame).
///
/// # Safety
/// - `buffers.0` must outlive the dispatch and match the kernel's slot sizes.
/// - `out_buf` must back `buffers.0[2]` (the dest).
/// - `user_params` must live across the call.
pub(crate) fn rayon_dispatch_tile<P: Copy + Sync>(
	width: u32,
	buffers: SafeBuffers,
	tp: FrameParams,
	user_params: &P,
	dispatch_tile_fn: CpuDispatchTileFn,
	out_buf: &mut [u8],
	out_stride_bytes: usize,
) -> u32 {
	use rayon::prelude::*;

	let buf_ptr = buffers.0.as_ptr() as usize;
	let tp_ptr = &tp as *const _ as usize;
	let up_ptr = user_params as *const _ as usize;

	let height = tp.height as usize;
	let rows_per_task = compute_rows_per_task(tp.height) as usize;
	let chunk_bytes = rows_per_task * out_stride_bytes;

	crate::cpu::pool::ensure_initialized();
	out_buf.par_chunks_mut(chunk_bytes).enumerate().for_each(move |(chunk_idx, _chunk_bytes)| {
		let y0 = (chunk_idx * rows_per_task) as u32;
		let y1 = ((chunk_idx * rows_per_task + rows_per_task).min(height)) as u32;
		unsafe {
			dispatch_tile_fn(
				y0,
				y1,
				width,
				buf_ptr as *const *const c_void,
				tp_ptr as *const c_void,
				up_ptr as *const c_void,
			);
		}
	});

	rows_per_task as u32
}

/// Dispatch a CPU kernel from a `Configuration` with no AE/Premiere plumbing.
///
/// Same code path as the Premiere render route minus the AE fallback; output is
/// partitioned at `dest_pitch_px * bytes_per_pixel` rows starting at `dest_data`.
///
/// # Safety
/// All pointers in `config` must be valid, non-aliasing where the kernel expects,
/// and live for the call.
pub unsafe fn render_cpu_direct<P: Copy + Sync>(
	kernel_name: &'static str,
	config: &Configuration,
	dispatch_tile_fn: CpuDispatchTileFn,
	user_params: &P,
) {
	use crate::cpu::diag;

	let w = config.width;
	let h = config.height;
	if w == 0 || h == 0 {
		return;
	}

	let guard = diag::DispatchGuard::enter();
	let wall_start = std::time::Instant::now();

	let outgoing_ptr = config.outgoing_data.unwrap_or(std::ptr::null_mut()) as *const c_void;
	let incoming_ptr = config.incoming_data.unwrap_or(std::ptr::null_mut()) as *const c_void;
	let dest_ptr = config.dest_data as *const c_void;

	let buffers = SafeBuffers([outgoing_ptr, incoming_ptr, dest_ptr]);

	let tp = FrameParams::from_config(config);

	let out_stride_bytes = tp.dst_desc.pitch_bytes as usize;
	let out_buf_size = (h as usize) * out_stride_bytes;

	let setup_ns = wall_start.elapsed().as_nanos() as u64;
	let body_start = std::time::Instant::now();
	let mut chunk_rows = 1u32;

	if out_buf_size > 0 && !dest_ptr.is_null() {
		// SAFETY: caller guarantees `dest_ptr` covers `out_buf_size` bytes; the slice is only used to partition rows across rayon workers.
		let out_buf = unsafe { std::slice::from_raw_parts_mut(dest_ptr as *mut u8, out_buf_size) };
		chunk_rows = rayon_dispatch_tile(w, buffers, tp, user_params, dispatch_tile_fn, out_buf, out_stride_bytes);
	}

	let body_ns = body_start.elapsed().as_nanos() as u64;
	crate::timing::record(kernel_name, crate::types::Backend::Cpu, setup_ns + body_ns);
	diag::log_dispatch(kernel_name, diag::DispatchPath::Direct, w, h, chunk_rows, setup_ns, body_ns, guard.concurrent_at_entry());
	drop(guard);
}
