use std::ffi::c_void;

use after_effects::log;

/// Metal command buffers already synchronize via `waitUntilCompleted` before
/// returning from `run()`. This is a no-op for API parity with the CUDA backend.
///
/// # Safety
/// `_queue` must be a valid Metal command queue handle.
pub unsafe fn sync_after_dispatch(_queue: *mut c_void, _generation: u64) -> Result<f32, &'static str> {
	Ok(0.0)
}

/// No-op on Metal.
///
/// # Safety
/// No preconditions.
pub unsafe fn cleanup() {
	log::info!("[Metal] Stream fences cleared");
}
