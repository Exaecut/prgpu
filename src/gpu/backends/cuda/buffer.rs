use cudarc::driver::sys::{cuCtxSetCurrent, cuMemAlloc_v2, cuMemFree_v2, CUcontext, CUdeviceptr, CUresult};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, BufferKey, BufferObj, ImageBuffer};
use crate::DeviceHandleInit;
use after_effects::log;

const MAX_GPU_BUFFER_ENTRIES: usize = 12;

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
	/// Returns the evicted `BufferObj` if an eviction occurred (caller must free it).
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

/// # Safety
/// `device` must be a valid CUcontext pointer.
pub(crate) unsafe fn allocate(device: *mut c_void, length_bytes: u64) -> *mut c_void {
	let ctx = device as CUcontext;
	unsafe { cuCtxSetCurrent(ctx) };

	let mut devptr: CUdeviceptr = 0;
	let result = unsafe { cuMemAlloc_v2(&mut devptr, length_bytes as usize) };

	match result {
		CUresult::CUDA_SUCCESS => devptr as *mut c_void,
		err => {
			log::error!("[CUDA] cuMemAlloc_v2 failed: {:?} (requested {} bytes)", err, length_bytes);
			std::ptr::null_mut()
		}
	}
}

/// Free a GPU buffer and log the result.
unsafe fn free_buffer(buf: BufferObj) {
	if !buf.raw.is_null() {
		let devptr = buf.raw as CUdeviceptr;
		let res = unsafe { cuMemFree_v2(devptr) };
		if res != CUresult::CUDA_SUCCESS {
			log::error!("[CUDA/buffer] cuMemFree_v2 failed during LRU eviction: {:?}", res);
		}
	}
}

/// # Safety
/// `device` must be a valid CUcontext pointer (FromPtr) or valid suite handle (FromSuite).
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

	// Cache miss - allocate new buffer
	let length = compute_length_bytes(width, height, bytes_per_pixel);
	let raw = match device {
		DeviceHandleInit::FromPtr(device) => unsafe { allocate(device, length) },
		DeviceHandleInit::FromSuite((device_index, suite)) => {
			suite.allocate_device_memory(device_index, length as usize).unwrap_or_else(|e| {
				log::error!("[CUDA] GPUDevice suite allocation failed: {e:?}");
				std::ptr::null_mut()
			})
		}
	};

	if raw.is_null() {
		log::error!("[CUDA/buffer] buffer allocation failed for {}x{} bpp={} tag={}", width, height, bytes_per_pixel, tag);
	}

	let obj = BufferObj { raw };
	let evicted = guard.insert(key, obj);

	// Drop the lock before freeing evicted memory (no need to hold it during GPU free)
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

/// # Safety
/// No GPU work may reference these buffers.
pub unsafe fn cleanup() {
	if let Some(cache) = CACHE.get() {
		let mut guard = cache.lock();
		for (_key, buf) in guard.entries.drain(..) {
			if !buf.raw.is_null() {
				let devptr = buf.raw as CUdeviceptr;
				let res = unsafe { cuMemFree_v2(devptr) };
				if res != CUresult::CUDA_SUCCESS {
					log::error!("[CUDA/buffer] cuMemFree_v2 failed during cleanup: {:?}", res);
				}
			}
		}
	}
}
