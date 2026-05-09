use std::ffi::c_void;

use after_effects::log;
use cudarc::driver::sys as cuda;

/// Block until enqueued GPU work on `stream` completes.
///
/// Adobe's buffer lifecycle requires inputs to be fully consumed before
/// `render()` returns; this guards against use-after-free.
///
/// # Safety: `stream` must be a valid CUDA stream with all passes enqueued.
pub unsafe fn sync_after_dispatch(stream: *mut c_void, _generation: u64) -> Result<f32, &'static str> {
	let res = unsafe { cuda::cuStreamSynchronize(stream as cuda::CUstream) };
	if res != cuda::CUresult::CUDA_SUCCESS {
		log::error!("[CUDA] cuStreamSynchronize failed: {:?}", res);
		return Err("cuStreamSynchronize failed");
	}
	Ok(0.0)
}

/// No-op; stream sync is stateless. # Safety: no GPU work in-flight.
pub unsafe fn cleanup() {}
