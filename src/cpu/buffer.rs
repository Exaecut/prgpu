use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::types::{BufferObj, ImageBuffer, compute_length_bytes, compute_row_bytes};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
	width: u32,
	height: u32,
	bytes_per_pixel: u32,
	tag: u32,
}

static CACHE: OnceLock<Mutex<HashMap<Key, Vec<u8>>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<Key, Vec<u8>>> {
	CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Returns a cached CPU heap buffer, allocating on first request for a given
/// `(width, height, bpp, tag)` combination. The returned `ImageBuffer.buf.raw`
/// pointer is valid until `cleanup()` is called.
pub fn get_or_create(
	width: u32,
	height: u32,
	bytes_per_pixel: u32,
	tag: u32,
) -> ImageBuffer {
	let key = Key { width, height, bytes_per_pixel, tag };
	let mut guard = cache().lock();

	let data = guard.entry(key).or_insert_with(|| {
		let len = compute_length_bytes(width, height, bytes_per_pixel) as usize;
		vec![0u8; len]
	});

	ImageBuffer {
		buf: BufferObj { raw: data.as_mut_ptr() as *mut c_void },
		width,
		height,
		bytes_per_pixel,
		row_bytes: compute_row_bytes(width, bytes_per_pixel),
		pitch_px: width,
	}
}

/// Frees all cached CPU buffers.
pub fn cleanup() {
	if let Some(map) = CACHE.get() {
		map.lock().drain();
	}
}
