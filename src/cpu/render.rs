use std::ffi::c_void;

use after_effects as ae;

use crate::types::{Configuration, FrameParams};

/// Per-pixel CPU dispatch: kernel is invoked once per `(x, y)` tuple.
/// Kept for the AE `iterate_with` path that drives `(x, y)` externally.
pub type CpuDispatchFn = unsafe extern "C" fn(u32, u32, *const *const c_void, *const c_void, *const c_void);

/// Tile CPU dispatch: kernel is invoked over a `[y0, y1) × [0, width)` range
/// inside a single C-side loop. Used by the rayon path so a single FFI
/// boundary amortizes over an entire tile — roughly `rows_per_task × width`
/// kernel invocations become one extern "C" call.
pub type CpuDispatchTileFn = unsafe extern "C" fn(u32, u32, u32, *const *const c_void, *const c_void, *const c_void);

/// Wrapper to make buffer pointer array `Send + Sync`.
///
/// SAFETY: The buffer pointers are valid for the duration of the dispatch call.
/// They point to Layer-backed or intermediate buffers that outlive the iteration.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SafeBuffers(pub(crate) [*const c_void; 3]);
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

	// Start global wall-clock + concurrency tracking as soon as we enter the
	// dispatcher. `setup_ns` covers everything before the rayon / AE loop
	// body; `body_ns` is the loop itself.
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

	let tp = FrameParams {
		out_desc: crate::types::make_texture_desc(w, h, config.outgoing_pitch_px as u32, config.bytes_per_pixel, config.pixel_layout),
		in_desc: crate::types::make_texture_desc(w, h, config.incoming_pitch_px as u32, config.bytes_per_pixel, config.pixel_layout),
		dst_desc: crate::types::make_texture_desc(w, h, config.dest_pitch_px as u32, config.bytes_per_pixel, config.pixel_layout),
		width: w,
		height: h,
		time,
		progress: config.progress,
	};

	let can_iterate_with = !in_data.is_premiere() && w == out_layer.width() as u32 && h == out_layer.height() as u32;

	let setup_ns = wall_start.elapsed().as_nanos() as u64;
	let body_start = std::time::Instant::now();

	let (path, chunk_rows, result) = if can_iterate_with {
		// AE iterate_with drives (x, y) externally, so we must use the
		// per-pixel entry point.
		(
			diag::DispatchPath::AeIterate,
			1u32,
			ae_dispatch(in_layer, out_layer, buffers, tp, user_params, dispatch_fn),
		)
	} else {
		// Rayon fallback — use the tile entry point (one FFI call per chunk).
		let out_stride_bytes = tp.dst_desc.pitch_bytes as usize;
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


/// Compute how many rows of work each rayon task should own.
///
/// The default strategy aims for ~4 tasks per worker thread — enough for good
/// load balancing but coarse enough to amortize rayon's fork-join overhead
/// across the per-pixel inner loop. For a 1516-row frame on 16 threads we go
/// from 1516 tasks (one per row) down to ~64, cutting scheduler work by ~24×.
#[inline]
fn compute_rows_per_task(height: u32) -> u32 {
	// Base chunking on the bounded render pool's worker count, not the global
	// rayon pool, so task granularity matches the pool we actually dispatch on.
	let threads = crate::cpu::pool::worker_count().max(1) as u32;
	let target_tasks = threads.saturating_mul(4).max(1);
	((height + target_tasks - 1) / target_tasks).max(1)
}

/// AE-free rayon tile dispatcher. Shared by the host-side render path
/// (Premiere / non-AE AE path) and the `prgpu::bench` harness.
///
/// Calls `dispatch_tile_fn` **once per rayon chunk**; the C-side wrapper
/// loops over the chunk's `[y0, y1) × [0, width)` internally. This
/// eliminates the per-pixel FFI boundary that, on Windows plugin DLLs with
/// dynamic-TLS, was costing ~100 ns/pixel (~350 ms per 3.57 Mpx frame).
///
/// Returns `rows_per_task`, i.e. the chunk granularity used by rayon; the
/// caller forwards this to the per-dispatch diagnostic.
///
/// # Safety considerations
/// - `buffers.0` must outlive the dispatch and point to memory of the sizes
///   implied by `tp` and the kernel's buffer slots.
/// - `out_buf` must be the backing storage of `buffers.0[2]` (the dest buffer);
///   it is used purely to partition row iteration across rayon workers.
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

/// Dispatch a CPU kernel without any After Effects / Premiere plumbing.
///
/// Used by the `prgpu::bench` harness and any host-side driver that already has
/// raw buffer pointers in a [`Configuration`]. This is the same code path as
/// the Premiere render route (pure rayon), minus the AE fallback.
///
/// The output buffer partitioning uses `config.dest_pitch_px * config.bytes_per_pixel`
/// as row stride and assumes the dest buffer contains `config.height` rows
/// starting at `config.dest_data`.
///
/// # Safety
/// All pointers in `config` must be valid, non-aliasing where the kernel expects
/// them to be non-aliasing, and remain alive for the duration of the call.
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

	let tp = FrameParams {
		out_desc: crate::types::make_texture_desc(w, h, config.outgoing_pitch_px as u32, config.bytes_per_pixel, config.pixel_layout),
		in_desc: crate::types::make_texture_desc(w, h, config.incoming_pitch_px as u32, config.bytes_per_pixel, config.pixel_layout),
		dst_desc: crate::types::make_texture_desc(w, h, config.dest_pitch_px as u32, config.bytes_per_pixel, config.pixel_layout),
		width: w,
		height: h,
		time: config.time,
		progress: config.progress,
	};

	let out_stride_bytes = tp.dst_desc.pitch_bytes as usize;
	let out_buf_size = (h as usize) * out_stride_bytes;

	let setup_ns = wall_start.elapsed().as_nanos() as u64;
	let body_start = std::time::Instant::now();
	let mut chunk_rows = 1u32;

	if out_buf_size > 0 && !dest_ptr.is_null() {
		// SAFETY: dest_ptr points to `out_buf_size` bytes of writable memory by
		// caller contract; slice is used only to partition rows across rayon workers.
		let out_buf = unsafe { std::slice::from_raw_parts_mut(dest_ptr as *mut u8, out_buf_size) };
		chunk_rows = rayon_dispatch_tile(w, buffers, tp, user_params, dispatch_tile_fn, out_buf, out_stride_bytes);
	}

	let body_ns = body_start.elapsed().as_nanos() as u64;
	crate::timing::record(kernel_name, crate::types::Backend::Cpu, setup_ns + body_ns);
	diag::log_dispatch(kernel_name, diag::DispatchPath::Direct, w, h, chunk_rows, setup_ns, body_ns, guard.concurrent_at_entry());
	drop(guard);
}
