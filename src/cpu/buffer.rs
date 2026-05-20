use std::cell::RefCell;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, mip_buffer_size_bytes, BufferObj, ImageBuffer};

const ALLOC_GUARD_BYTES: usize = 64;
const MAX_CPU_BUFFER_ENTRIES: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
	width: u32,
	height: u32,
	bytes_per_pixel: u32,
	tag: u32,
	mip_levels: u32,
}

/// Ordered LRU: MRU at the back, LRU at the front. `MAX_CPU_BUFFER_ENTRIES <= 12` keeps the linear scan negligible.
struct OrderedLru {
	entries: Vec<(Key, Vec<u8>)>,
	capacity: usize,
}

impl OrderedLru {
	fn new(capacity: usize) -> Self {
		Self {
			entries: Vec::with_capacity(capacity),
			capacity,
		}
	}

	/// Promote `key` to MRU; returns true on hit.
	fn promote(&mut self, key: &Key) -> bool {
		if let Some(idx) = self.entries.iter().position(|(k, _)| k == key) {
			let entry = self.entries.remove(idx);
			self.entries.push(entry);
			true
		} else {
			false
		}
	}

	/// Mutable pointer to the MRU entry. Only valid right after `promote` returned true or after `insert`.
	fn last_data_ptr(&mut self) -> *mut c_void {
		self.entries.last_mut().unwrap().1.as_mut_ptr() as *mut c_void
	}

	/// Insert, evicting LRU when at capacity. Returns the evicted (key, len).
	fn insert(&mut self, key: Key, value: Vec<u8>) -> Option<(Key, usize)> {
		let evicted = if self.entries.len() >= self.capacity {
			let (k, v) = self.entries.remove(0);
			Some((k, v.len()))
		} else {
			None
		};
		self.entries.push((key, value));
		evicted
	}

	fn clear(&mut self) {
		self.entries.clear();
	}
}

thread_local! {
	static CPU_CACHE: RefCell<OrderedLru> = RefCell::new(OrderedLru::new(MAX_CPU_BUFFER_ENTRIES));
}

pub fn get_or_create(width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> ImageBuffer {
	get_or_create_with_mips(width, height, bytes_per_pixel, 1, tag)
}

/// Cache-aware variant: returns `(buffer, was_hit)`. Callers that need to
/// populate the buffer only on first allocation (e.g. source snapshot) use
/// `was_hit` to skip the upload on cache hit. See `prepare_source_snapshot`.
pub fn get_or_create_returning_hit(width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> (ImageBuffer, bool) {
	get_or_create_with_mips_inner(width, height, bytes_per_pixel, 1, tag)
}

/// Like `get_or_create` but sizes the buffer for an `mip_levels`-deep mip chain.
/// `mip_levels <= 1` behaves identically; otherwise the byte budget follows
/// `mip_buffer_size_bytes` so the prgpu downsample pass writes through without re-alloc.
pub fn get_or_create_with_mips(width: u32, height: u32, bytes_per_pixel: u32, mip_levels: u32, tag: u32) -> ImageBuffer {
	get_or_create_with_mips_inner(width, height, bytes_per_pixel, mip_levels, tag).0
}

fn get_or_create_with_mips_inner(width: u32, height: u32, bytes_per_pixel: u32, mip_levels: u32, tag: u32) -> (ImageBuffer, bool) {
	let key = Key {
		width,
		height,
		bytes_per_pixel,
		tag,
		mip_levels: mip_levels.max(1),
	};

	CPU_CACHE.with(|cache| {
		let mut guard = cache.borrow_mut();

		if guard.promote(&key) {
			let raw = guard.last_data_ptr();
			return (
				ImageBuffer {
					buf: BufferObj { raw },
					width,
					height,
					bytes_per_pixel,
					row_bytes: compute_row_bytes(width, bytes_per_pixel),
					pitch_px: width,
				},
				true,
			);
		}

		let len = if mip_levels <= 1 {
			compute_length_bytes(width, height, bytes_per_pixel) as usize
		} else {
			mip_buffer_size_bytes(width, height, bytes_per_pixel, mip_levels) as usize
		};
		let data = vec![0u8; len + ALLOC_GUARD_BYTES];

		guard.insert(key, data);
		let raw = guard.last_data_ptr();

		(
			ImageBuffer {
				buf: BufferObj { raw },
				width,
				height,
				bytes_per_pixel,
				row_bytes: compute_row_bytes(width, bytes_per_pixel),
				pitch_px: width,
			},
			false,
		)
	})
}

pub fn cleanup() {
	CPU_CACHE.with(|cache| {
		cache.borrow_mut().clear();
	});
}
