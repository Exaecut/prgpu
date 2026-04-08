use std::ffi::c_void;

use after_effects::log;
use cudarc::driver::sys as cuda;

/// Blocks until all enqueued GPU work on the stream completes.
///
/// Adobe's buffer lifecycle requires input buffers to be fully consumed
/// before `render()` returns. This sync ensures no use-after-free.
///
/// # Safety
/// `stream` must be a valid CUDA stream with all passes already enqueued.
pub unsafe fn sync_after_dispatch(
    stream: *mut c_void,
    _generation: u64,
) -> Result<f32, &'static str> {
    let res = unsafe { cuda::cuStreamSynchronize(stream as cuda::CUstream) };
    if res != cuda::CUresult::CUDA_SUCCESS {
        log::error!("[CUDA] cuStreamSynchronize failed: {:?}", res);
        return Err("cuStreamSynchronize failed");
    }
    Ok(0.0)
}

/// No-op — stream sync is stateless (no cached events to destroy).
///
/// # Safety
/// No GPU work may be in-flight.
pub unsafe fn cleanup() {}
