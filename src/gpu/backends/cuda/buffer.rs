use std::{collections::HashMap, ffi::c_void};
use std::sync::OnceLock;
use parking_lot::Mutex;

/// Key that uniquely identifies a cached GPU buffer allocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferKey {
	/// Device pointer (CUDevice*) used for allocation - cast to usize for hashing
	pub device: usize,
	/// Width in pixels (for convenience when used as an image buffer)
	pub width: u32,
	/// Height in pixels
	pub height: u32,
	/// Bytes per pixel (e.g. 16 for float4, 8 for half4)
	pub bytes_per_pixel: u32,
	/// Optional tag to differentiate multiple buffers of the same size (0 by default)
	pub tag: u32,
}

/// Thin wrapper around an CUDA Buffer that we explicitly mark Send + Sync.
/// You are responsible for lifetime via `cleanup()`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BufferObj {
	pub raw: *mut c_void,
}

// We commonly pass CUDA Buffers across threads in render pipelines.
// Marking these wrappers as Send/Sync is a deliberate design choice here.
unsafe impl Send for BufferObj {}
unsafe impl Sync for BufferObj {}

/// Public view returned to callers. Copying is cheap and does not affect ownership.
/// The underlying buffer is owned by the cache and freed by `cleanup()`.
#[derive(Clone, Copy)]
pub struct ImageBuffer {
	pub buf: BufferObj,
	pub width: u32,
	pub height: u32,
	/// Bytes per pixel
	pub bytes_per_pixel: u32,
	/// Row bytes in bytes (for APIs that want bytes)
	pub row_bytes: u32,
	/// Pitch in pixels (what your shaders use)
	pub pitch_px: u32,
}

// Internal cache: one buffer per (device, size, bpp, tag).
static CACHE: OnceLock<Mutex<HashMap<BufferKey, BufferObj>>> = OnceLock::new();

#[inline]
fn compute_row_bytes(width: u32, bytes_per_pixel: u32) -> u32 {
	width.saturating_mul(bytes_per_pixel)
}

#[inline]
fn compute_length_bytes(width: u32, height: u32, bytes_per_pixel: u32) -> u64 {
	(width as u64) * (height as u64) * (bytes_per_pixel as u64)
}

pub unsafe fn create_raw_buffer(device: *mut Object, length_bytes: u64) -> *mut Object {
	todo!("Implement raw buffer creation for CUDA backend");
}

/// Create an "image-like" buffer sized width*height with the given bytes_per_pixel.
/// 
/// # Safety
/// - `device` must be a valid pointer to an CUDevice*.
/// - The caller must ensure that the returned buffer is properly managed and released when no longer needed.
pub unsafe fn create_texture_buffer(device: *mut Object, width: u32, height: u32, bytes_per_pixel: u32) -> *mut Object {
	let length = compute_length_bytes(width, height, bytes_per_pixel);
	unsafe { create_raw_buffer(device, length) }
}

/// Get a cached buffer or create-and-cache one if absent.
/// Returns an `ImageBuffer` view with useful stride info populated.
/// 
/// # Safety
/// - `device` must be a valid pointer to an CUDevice*.
/// - The caller must ensure that the returned buffer is properly managed and released when no longer needed.
pub unsafe fn get_or_create(device: *mut c_void, width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> ImageBuffer {
	let key = BufferKey {
		device: device as usize,
		width,
		height,
		bytes_per_pixel,
		tag,
	};

	let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
	let mut guard = map.lock();

	let buf = if let Some(existing) = guard.get(&key) {
		*existing
	} else {
		let raw = unsafe { create_texture_buffer(device, width, height, bytes_per_pixel) };
		let obj = BufferObj { raw };
		guard.insert(key, obj);
		obj
	};

	let row_bytes = compute_row_bytes(width, bytes_per_pixel);
	let pitch_px = width;

	ImageBuffer {
		buf,
		width,
		height,
		bytes_per_pixel,
		row_bytes,
		pitch_px,
	}
}

pub unsafe fn cleanup() {
	todo!("Implement clean for CUDA backend");
}
