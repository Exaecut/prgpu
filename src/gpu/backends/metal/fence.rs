use std::ffi::c_void;

use after_effects::log;

/// No-op for API parity with CUDA; Metal command buffers already sync via `waitUntilCompleted` before `run()` returns.
///
/// # Safety: `_queue` must be a valid Metal command queue.
pub unsafe fn sync_after_dispatch(_queue: *mut c_void, _generation: u64) -> Result<f32, &'static str> {
	Ok(0.0)
}

/// No-op on Metal. # Safety: no preconditions.
pub unsafe fn cleanup() {
	log::info!("[Metal] Stream fences cleared");
}
