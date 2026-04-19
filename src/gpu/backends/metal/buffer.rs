use std::sync::OnceLock;

use objc::{msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use crate::types::{compute_length_bytes, compute_row_bytes, BufferKey, BufferObj, ImageBuffer};
use crate::DeviceHandleInit;

const MAX_GPU_BUFFER_ENTRIES: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageMode {
	#[allow(dead_code)]
	Shared = 0,
	Private = 2,
}

impl StorageMode {
	fn as_resource_options(self) -> u64 {
		(self as u64) << 4
	}
}

/// Simple ordered LRU cache: most-recently-used at the back, LRU at the front.
/// With `MAX_GPU_BUFFER_ENTRIES <= 12`, linear scan is negligible.
struct OrderedLru {
	entries: Vec<(BufferKey, BufferObj)>,
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
	/// Returns the `BufferObj` copy if found, `None` otherwise.
	fn get(&mut self, key: &BufferKey) -> Option<BufferObj> {
		if let Some(idx) = self.entries.iter().position(|(k, _)| k == key) {
			let entry = self.entries.remove(idx);
			self.entries.push(entry);
			Some(self.entries.last().unwrap().1)
		} else {
			None
		}
	}

	/// Insert a new entry, evicting LRU if at capacity.
	/// Returns the evicted `BufferObj` if an eviction occurred (caller must release it).
	fn insert(&mut self, key: BufferKey, value: BufferObj) -> Option<BufferObj> {
		let evicted = if self.entries.len() >= self.capacity {
			// Evict LRU (front)
			let (_, v) = self.entries.remove(0);
			Some(v)
		} else {
			None
		};
		self.entries.push((key, value));
		evicted
	}

}

static CACHE: OnceLock<Mutex<OrderedLru>> = OnceLock::new();

fn cache() -> &'static Mutex<OrderedLru> {
	CACHE.get_or_init(|| Mutex::new(OrderedLru::new(MAX_GPU_BUFFER_ENTRIES)))
}

pub(crate) unsafe fn allocate(device: *mut Object, length_bytes: u64) -> *mut Object {
	let opts = StorageMode::Private.as_resource_options();
	msg_send![device, newBufferWithLength: length_bytes options: opts]
}

/// Free a Metal buffer by sending `[release]`.
unsafe fn free_buffer(buf: BufferObj) {
	if !buf.raw.is_null() {
		let _: () = msg_send![buf.raw as *mut Object, release];
	}
}

pub unsafe fn get_or_create(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> ImageBuffer {
	let key = match device {
		DeviceHandleInit::FromPtr(device) => BufferKey {
			device: device as usize,
			width,
			height,
			bytes_per_pixel,
			tag,
		},
		DeviceHandleInit::FromSuite((device_index, suite)) => {
			let device_handle = suite.device_info(device_index).map(|info| info.outDeviceHandle as usize).unwrap_or(0);
			BufferKey {
				device: device_handle,
				width,
				height,
				bytes_per_pixel,
				tag,
			}
		}
	};

	let mut guard = cache().lock();

	// Try cache hit first - promote to MRU
	if let Some(existing) = guard.get(&key) {
		return ImageBuffer {
			buf: existing,
			width,
			height,
			bytes_per_pixel,
			row_bytes: compute_row_bytes(width, bytes_per_pixel),
			pitch_px: width,
		};
	}

	// Cache miss
	let raw = match device {
		DeviceHandleInit::FromPtr(device) => {
			let length = compute_length_bytes(width, height, bytes_per_pixel);
			unsafe { allocate(device as *mut Object, length) as *mut std::ffi::c_void }
		}
		DeviceHandleInit::FromSuite((device_index, suite)) => {
			let length = compute_length_bytes(width, height, bytes_per_pixel) as usize;
			unsafe { suite.allocate_device_memory(device_index, length) }.unwrap_or_else(|e| {
				after_effects::log::error!("[Metal] GPUDevice suite allocation failed: {e:?}");
				std::ptr::null_mut()
			})
		}
	};

	let obj = BufferObj { raw };
	let evicted = guard.insert(key, obj);

	// Drop the lock before releasing evicted memory
	drop(guard);

	if let Some(evicted_buf) = evicted {
		unsafe { free_buffer(evicted_buf) };
	}

	ImageBuffer {
		buf: BufferObj { raw },
		width,
		height,
		bytes_per_pixel,
		row_bytes: compute_row_bytes(width, bytes_per_pixel),
		pitch_px: width,
	}
}

pub unsafe fn cleanup() {
	if let Some(cache) = CACHE.get() {
		let mut guard = cache.lock();
		for (_, b) in guard.entries.drain(..) {
			if !b.raw.is_null() {
				let _: () = msg_send![b.raw as *mut Object, release];
			}
		}
	}
}
