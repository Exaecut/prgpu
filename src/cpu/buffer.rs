use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, BufferObj, ImageBuffer};

/// Extra bytes appended past the nominal end of every CPU intermediate buffer.
///
/// # Why this exists
///
/// Sequential same-sized `Vec::new` calls on Windows place the heap block
/// header of `buf_b` exactly 16 bytes after the last usable byte of `buf_a`.
/// If any pixel kernel writes one slot past the last valid index (e.g. an
/// off-by-one in a loop bound, or an index computed before `clamp_xy` is
/// applied), that write silently overwrites the `buf_b` block header.  The
/// Windows heap then raises **`c0000374`** (heap corruption) on the next
/// allocation, followed by a **`c0000005`** access-violation when the
/// corrupted free-list pointer is dereferenced.
///
/// 64 bytes = 4 × RGBA-32f pixels, which covers the widest possible
/// single-pixel overrun.  The `ImageBuffer` returned still reports the
/// nominal `pitch_px` and `row_bytes`, so all kernel index math is
/// unaffected; the guard region merely acts as a silent buffer zone.
const ALLOC_GUARD_BYTES: usize = 64;

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
			// Allocate with ALLOC_GUARD_BYTES extra to prevent heap corruption
			// when an OOB write of up to one max-format pixel past the nominal
			// end would otherwise overwrite the adjacent heap block's header.
			vec![0u8; len + ALLOC_GUARD_BYTES]
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

pub fn cleanup() {
	CPU_CACHE.with(|cache| {
		cache.borrow_mut().clear();
	});
}
