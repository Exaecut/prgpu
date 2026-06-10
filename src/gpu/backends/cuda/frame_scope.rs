//! Per-frame CUDA submission scope.
//!
//! Batches every pass of a frame on one stream with zero per-pass
//! allocation and a single sync. The adapter brackets
//! a frame with [`begin`]/[`end`]; while the scope is active, `cuda::run`
//! skips `cuCtxSetCurrent` and `cuStreamSynchronize`, and stages kernel
//! params in a persistent per-context device arena via `cuMemcpyHtoDAsync`
//! instead of `cuMemAlloc`+`cuMemcpyHtoD`+`cuMemFree`.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::OnceLock;

use after_effects::log;
use cudarc::driver::sys::{self as cuda, CUdeviceptr, CUresult};
use parking_lot::Mutex;

use crate::types::FrameScopeDesc;

/// Never returned on CUDA; exists for facade parity with the Metal scope,
/// whose frame command buffer can be killed by the macOS GPU watchdog.
pub const ERR_WATCHDOG: &str = "metal frame watchdog";

// 12-pass pyramid uses ~12 x (512 + 256) bytes; 256 KiB leaves generous slack
// for transitions and future multi-input graphs before the per-pass fallback.
const ARENA_CAPACITY: usize = 256 * 1024;
const ARENA_ALIGN: usize = 256;

// Keyed by (thread, ctx): Premiere can call render() concurrently from
// several threads sharing one CUcontext, and a begin() on one thread must not
// reset the cursor under another thread's in-flight frame.
struct Arena {
	thread: std::thread::ThreadId,
	ctx: usize,
	base: CUdeviceptr,
	cursor: usize,
}

static ARENAS: OnceLock<Mutex<Vec<Arena>>> = OnceLock::new();

fn arenas() -> &'static Mutex<Vec<Arena>> {
	ARENAS.get_or_init(|| Mutex::new(Vec::new()))
}

#[derive(Clone, Copy)]
struct Scope {
	active: bool,
	ctx: *mut c_void,
	stream: *mut c_void,
	passes: u32,
	arena_misses: u32,
	ev_start: cuda::CUevent,
	ev_end: cuda::CUevent,
}

impl Scope {
	const fn inactive() -> Self {
		Self {
			active: false,
			ctx: std::ptr::null_mut(),
			stream: std::ptr::null_mut(),
			passes: 0,
			arena_misses: 0,
			ev_start: std::ptr::null_mut(),
			ev_end: std::ptr::null_mut(),
		}
	}
}

thread_local! {
	static SCOPE: Cell<Scope> = const { Cell::new(Scope::inactive()) };
}

/// Enter the frame scope: set the CUDA context current once for the whole
/// frame and reset the param arena cursor. No-op when the descriptor carries
/// no CUDA context (CPU/test paths).
pub fn begin(desc: &FrameScopeDesc) {
	let Some(ctx) = desc.context_handle else { return };
	if ctx.is_null() || desc.command_queue_handle.is_null() {
		return;
	}
	let res = unsafe { cuda::cuCtxSetCurrent(ctx as cuda::CUcontext) };
	if res != CUresult::CUDA_SUCCESS {
		log::error!("[CUDA/frame] cuCtxSetCurrent failed at frame begin: {res:?}");
		return;
	}
	{
		let tid = std::thread::current().id();
		let mut guard = arenas().lock();
		if let Some(a) = guard.iter_mut().find(|a| a.thread == tid && a.ctx == ctx as usize) {
			a.cursor = 0;
		}
	}
	// Frame timing via cuEvent pair: GPU-side elapsed ms, comparable to the
	// Metal GPUStartTime/GPUEndTime.
	let mut ev_start: cuda::CUevent = std::ptr::null_mut();
	let mut ev_end: cuda::CUevent = std::ptr::null_mut();
	unsafe {
		let flags = cuda::CUevent_flags_enum::CU_EVENT_DEFAULT as u32;
		if cuda::cuEventCreate(&mut ev_start, flags) != CUresult::CUDA_SUCCESS || cuda::cuEventCreate(&mut ev_end, flags) != CUresult::CUDA_SUCCESS {
			ev_start = std::ptr::null_mut();
			ev_end = std::ptr::null_mut();
		} else {
			cuda::cuEventRecord(ev_start, desc.command_queue_handle as cuda::CUstream);
		}
	}

	SCOPE.with(|s| {
		s.set(Scope {
			active: true,
			ctx,
			stream: desc.command_queue_handle,
			passes: 0,
			arena_misses: 0,
			ev_start,
			ev_end,
		})
	});
}

/// Leave the frame scope and block until every enqueued pass completes.
/// The one sync per frame Adobe's buffer lifecycle requires.
pub fn end(desc: &FrameScopeDesc) -> Result<(), &'static str> {
	let scope = SCOPE.with(|s| s.replace(Scope::inactive()));
	if !scope.active {
		return Ok(());
	}
	let stream = if scope.stream.is_null() { desc.command_queue_handle } else { scope.stream };
	if !scope.ev_end.is_null() {
		unsafe { cuda::cuEventRecord(scope.ev_end, stream as cuda::CUstream) };
	}
	let res = unsafe { cuda::cuStreamSynchronize(stream as cuda::CUstream) };

	let mut gpu_ms = -1.0f32;
	if !scope.ev_start.is_null() && !scope.ev_end.is_null() {
		unsafe {
			cuda::cuEventElapsedTime_v2(&mut gpu_ms, scope.ev_start, scope.ev_end);
			cuda::cuEventDestroy_v2(scope.ev_start);
			cuda::cuEventDestroy_v2(scope.ev_end);
		}
		crate::timing::record("frame", crate::types::Backend::Cuda, (gpu_ms.max(0.0) * 1_000_000.0) as u64);
	}
	log::debug!(
		"[CUDA/frame] gen={} backend=cuda gpu_ms={gpu_ms:.3} passes={} stream_syncs=1 param_arena_misses={}",
		desc.render_generation,
		scope.passes,
		scope.arena_misses
	);
	if res != CUresult::CUDA_SUCCESS {
		log::error!("[CUDA/frame] cuStreamSynchronize failed at frame end: {res:?}");
		return Err("frame-end cuStreamSynchronize failed");
	}
	Ok(())
}

pub(crate) fn is_active() -> bool {
	SCOPE.with(|s| s.get().active)
}

pub(crate) fn stream() -> *mut c_void {
	SCOPE.with(|s| s.get().stream)
}

pub(crate) fn note_pass() {
	SCOPE.with(|s| {
		let mut v = s.get();
		if v.active {
			v.passes += 1;
			s.set(v);
		}
	});
}

fn note_arena_miss() {
	SCOPE.with(|s| {
		let mut v = s.get();
		if v.active {
			v.arena_misses += 1;
			s.set(v);
		}
	});
}

/// Stage `bytes` into the per-context param arena with an async H2D on the
/// scope stream. Returns the device pointer, or `None` when the scope is
/// inactive or the arena is exhausted (caller falls back to alloc/free).
///
/// `cuMemcpyHtoDAsync` from pageable host memory returns only after the bytes
/// are staged, so stack-resident params are safe to drop after this call.
pub(crate) fn stage_params(bytes: &[u8]) -> Option<CUdeviceptr> {
	let scope = SCOPE.with(|s| s.get());
	if !scope.active {
		return None;
	}
	let ctx_key = scope.ctx as usize;
	let tid = std::thread::current().id();
	let size = bytes.len().div_ceil(ARENA_ALIGN) * ARENA_ALIGN;

	let mut guard = arenas().lock();
	let arena = match guard.iter_mut().position(|a| a.thread == tid && a.ctx == ctx_key) {
		Some(i) => &mut guard[i],
		None => {
			let mut base: CUdeviceptr = 0;
			let res = unsafe { cuda::cuMemAlloc_v2(&mut base, ARENA_CAPACITY) };
			if res != CUresult::CUDA_SUCCESS {
				log::error!("[CUDA/frame] param arena allocation failed: {res:?}");
				note_arena_miss();
				return None;
			}
			log::debug!("[CUDA/frame] param arena created: {ARENA_CAPACITY} bytes for ctx {ctx_key:#x}");
			guard.push(Arena {
				thread: tid,
				ctx: ctx_key,
				base,
				cursor: 0,
			});
			guard.last_mut().unwrap()
		}
	};

	if arena.cursor + size > ARENA_CAPACITY {
		note_arena_miss();
		return None;
	}
	let dst = arena.base + arena.cursor as u64;
	arena.cursor += size;
	drop(guard);

	let res = unsafe { cuda::cuMemcpyHtoDAsync_v2(dst, bytes.as_ptr() as *const c_void, bytes.len(), scope.stream as cuda::CUstream) };
	if res != CUresult::CUDA_SUCCESS {
		log::error!("[CUDA/frame] cuMemcpyHtoDAsync_v2 failed: {res:?}");
		return None;
	}
	Some(dst)
}

/// # Safety: no GPU work may reference the arenas.
pub unsafe fn cleanup() {
	if let Some(m) = ARENAS.get() {
		let mut guard = m.lock();
		for a in guard.drain(..) {
			if a.base != 0 {
				unsafe { cuda::cuMemFree_v2(a.base) };
			}
		}
	}
}
