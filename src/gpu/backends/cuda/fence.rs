use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use after_effects::log;
use cudarc::driver::sys as cuda;
use parking_lot::Mutex;

struct StreamFence {
    event: cuda::CUevent,
    generation: u64,
}

unsafe impl Send for StreamFence {}
unsafe impl Sync for StreamFence {}

static FENCES: OnceLock<Mutex<HashMap<usize, StreamFence>>> = OnceLock::new();

fn fences() -> &'static Mutex<HashMap<usize, StreamFence>> {
    FENCES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ensure_event(guard: &mut HashMap<usize, StreamFence>, key: usize) -> bool {
    if guard.contains_key(&key) {
        return true;
    }

    let mut event: cuda::CUevent = std::ptr::null_mut();
    let res = unsafe { cuda::cuEventCreate(&mut event, 0) };
    if res != cuda::CUresult::CUDA_SUCCESS {
        log::error!("[CUDA] cuEventCreate for fence failed");
        return false;
    }

    guard.insert(key, StreamFence { event, generation: 0 });
    true
}

/// Records a fence event on the stream after all kernel passes, then blocks
/// until the GPU completes all enqueued work up to this point.
///
/// Adobe's buffer lifecycle requires input buffers to be fully consumed
/// before `render()` returns. This sync ensures no use-after-free.
///
/// Returns `(gpu_ms, generation)` — GPU elapsed time and the generation synced.
///
/// # Safety
/// `stream` must be a valid CUDA stream with all passes already enqueued.
pub unsafe fn sync_after_dispatch(
    stream: *mut c_void,
    generation: u64,
) -> Result<f32, &'static str> {
    let key = stream as usize;
    let mut guard = fences().lock();

    if !ensure_event(&mut guard, key) {
        return Err("fence init failed");
    }

    let fence = guard.get_mut(&key).unwrap();

    let res = unsafe { cuda::cuEventRecord(fence.event, stream as cuda::CUstream) };
    if res != cuda::CUresult::CUDA_SUCCESS {
        log::error!("[CUDA] cuEventRecord failed on stream {key:#x}");
        return Err("cuEventRecord failed");
    }

    let res = unsafe { cuda::cuEventSynchronize(fence.event) };
    if res != cuda::CUresult::CUDA_SUCCESS {
        log::error!("[CUDA] cuEventSynchronize failed on stream {key:#x}");
        return Err("cuEventSynchronize failed");
    }

    fence.generation = generation;
    Ok(0.0)
}

/// Destroys all stream fence events.
///
/// # Safety
/// No GPU work may be in-flight on fenced streams.
pub unsafe fn cleanup() {
    let mut guard = fences().lock();
    for (_key, fence) in guard.drain() {
        if !fence.event.is_null() {
            unsafe { cuda::cuEventDestroy_v2(fence.event) };
        }
    }
    log::info!("[CUDA] Stream fences cleared");
}
