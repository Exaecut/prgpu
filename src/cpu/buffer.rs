use std::cell::RefCell;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, BufferObj, ImageBuffer};

const ALLOC_GUARD_BYTES: usize = 64;
const MAX_CPU_BUFFER_ENTRIES: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
	width: u32,
	height: u32,
	bytes_per_pixel: u32,
	tag: u32,
}

/// Simple ordered LRU cache: most-recently-used at the back, LRU at the front.
/// With `MAX_CPU_BUFFER_ENTRIES <= 12`, linear scan is negligible.
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

	/// Promote an existing entry to MRU position (back of the vector).
	/// Returns `true` if the key was found and promoted.
	fn promote(&mut self, key: &Key) -> bool {
		if let Some(idx) = self.entries.iter().position(|(k, _)| k == key) {
			let entry = self.entries.remove(idx);
			self.entries.push(entry);
			true
		} else {
			false
		}
	}

	/// Get a mutable pointer to the data of the MRU entry (last in vector).
	/// Only valid to call after `promote` returned `true` or immediately after `insert`.
	fn last_data_ptr(&mut self) -> *mut c_void {
		self.entries.last_mut().unwrap().1.as_mut_ptr() as *mut c_void
	}

	/// Insert a new entry, evicting LRU if at capacity.
	/// Returns evicted entry info if an eviction occurred.
	fn insert(&mut self, key: Key, value: Vec<u8>) -> Option<(Key, usize)> {
		let evicted = if self.entries.len() >= self.capacity {
			// Evict LRU (front)
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
	let key = Key {
		width,
		height,
		bytes_per_pixel,
		tag,
	};

	CPU_CACHE.with(|cache| {
		let mut guard = cache.borrow_mut();

		// Try cache hit first — promote to MRU
		if guard.promote(&key) {
			let raw = guard.last_data_ptr();
			return ImageBuffer {
				buf: BufferObj { raw },
				width,
				height,
				bytes_per_pixel,
				row_bytes: compute_row_bytes(width, bytes_per_pixel),
				pitch_px: width,
			};
		}

		// Cache miss — allocate new buffer
		let len = compute_length_bytes(width, height, bytes_per_pixel) as usize;
		let data = vec![0u8; len + ALLOC_GUARD_BYTES];

		guard.insert(key, data);
		let raw = guard.last_data_ptr();

		ImageBuffer {
			buf: BufferObj { raw },
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
