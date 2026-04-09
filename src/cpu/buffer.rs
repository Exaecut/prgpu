use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, BufferObj, ImageBuffer};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
	width: u32,
	height: u32,
	bytes_per_pixel: u32,
	tag: u32,
}

thread_local! {
	static CPU_CACHE: RefCell<HashMap<Key, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// Returns a cached CPU heap buffer, allocating on first request for a given
/// `(width, height, bpp, tag)` combination. The returned `ImageBuffer.buf.raw`
/// pointer is valid until `cleanup()` is called or the thread exits.
///
/// Thread-safe: each thread maintains its own independent cache.
pub fn get_or_create(width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> ImageBuffer {
	let key = Key {
		width,
		height,
		bytes_per_pixel,
		tag,
	};

	CPU_CACHE.with(|cache| {
		let mut guard = cache.borrow_mut();

		let data = guard.entry(key).or_insert_with(|| {
			let len = compute_length_bytes(width, height, bytes_per_pixel) as usize;
			vec![0u8; len]
		});

		ImageBuffer {
			buf: BufferObj {
				raw: data.as_mut_ptr() as *mut c_void,
			},
			width,
			height,
			bytes_per_pixel,
			row_bytes: compute_row_bytes(width, bytes_per_pixel),
			pitch_px: width,
		}
	})
}

/// Frees all cached CPU buffers for the current thread.
pub fn cleanup() {
	CPU_CACHE.with(|cache| {
		cache.borrow_mut().clear();
	});
}
