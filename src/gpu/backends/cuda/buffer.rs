use cudarc::driver::sys::{cuCtxSetCurrent, cuMemAlloc_v2, cuMemFree_v2, CUcontext, CUdeviceptr, CUresult};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, mip_buffer_size_bytes, BufferKey, BufferObj, ImageBuffer};
use crate::DeviceHandleInit;
use after_effects::log;

const MAX_GPU_BUFFER_ENTRIES: usize = 12;

/// Ordered LRU: MRU at the back, LRU at the front. `MAX_GPU_BUFFER_ENTRIES <= 12` keeps the linear scan negligible.
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

	/// Promote `key` to MRU; returns the `BufferObj` on hit, `None` otherwise.
	fn get(&mut self, key: &BufferKey) -> Option<BufferObj> {
		if let Some(idx) = self.entries.iter().position(|(k, _)| k == key) {
			let entry = self.entries.remove(idx);
			self.entries.push(entry);
			Some(self.entries.last().unwrap().1)
		} else {
			None
		}
	}

	/// Insert, evicting LRU when at capacity. Returns the evicted `BufferObj` (caller frees it).
	fn insert(&mut self, key: BufferKey, value: BufferObj) -> Option<BufferObj> {
		let evicted = if self.entries.len() >= self.capacity {
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

/// # Safety: `device` must be a valid CUcontext.
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

unsafe fn free_buffer(buf: BufferObj) {
	if !buf.raw.is_null() {
		let devptr = buf.raw as CUdeviceptr;
		let res = unsafe { cuMemFree_v2(devptr) };
		if res != CUresult::CUDA_SUCCESS {
			log::error!("[CUDA/buffer] cuMemFree_v2 failed during LRU eviction: {:?}", res);
		}
	}
}

/// # Safety: `device` must be a valid CUcontext (FromPtr) or suite handle (FromSuite).
pub unsafe fn get_or_create(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> ImageBuffer {
	unsafe { get_or_create_with_mips(device, width, height, bytes_per_pixel, 1, tag) }
}

/// Like `get_or_create` but sized for an `mip_levels`-deep mip chain.
///
/// # Safety: see `get_or_create`.
pub unsafe fn get_or_create_with_mips(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, mip_levels: u32, tag: u32) -> ImageBuffer {
	let mips = mip_levels.max(1);
	let key = match device {
		DeviceHandleInit::FromPtr(device) => BufferKey {
			device: device as usize,
			width,
			height,
			bytes_per_pixel,
			tag,
			mip_levels: mips,
		},
		DeviceHandleInit::FromSuite((device_index, suite)) => {
			let device_handle = suite.device_info(device_index).map(|info| info.outDeviceHandle as usize).unwrap_or(0);
			BufferKey {
				device: device_handle,
				width,
				height,
				bytes_per_pixel,
				tag,
				mip_levels: mips,
			}
		}
	};

	let mut guard = cache().lock();

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

	let length = if mips <= 1 {
		compute_length_bytes(width, height, bytes_per_pixel)
	} else {
		mip_buffer_size_bytes(width, height, bytes_per_pixel, mips) as u64
	};
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

	// Drop the lock before freeing evicted memory; no need to hold it across the GPU free.
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

/// Buffer-to-buffer device copy. `cuMemcpy2D_v2` for mismatched pitches
/// (Premiere's padded source vs. tight mip buffer); falls back to flat
/// `cuMemcpyDtoD_v2` when pitches match.
///
/// Synchronous on the default stream so subsequent dispatches see the copied data.
///
/// # Safety
/// - `device` must currently own both `src` and `dst`.
/// - Both must hold at least `pitch_bytes * height` bytes from their offsets.
/// - No other GPU work may touch `dst` concurrently.
pub unsafe fn copy_buffer(
	device: *mut c_void,
	src: *mut c_void,
	src_offset: u64,
	src_pitch_bytes: u32,
	dst: *mut c_void,
	dst_offset: u64,
	dst_pitch_bytes: u32,
	width_bytes: u32,
	height: u32,
) -> Result<(), &'static str> {
	use cudarc::driver::sys::{cuMemcpy2D_v2, cuMemcpyDtoD_v2, CUDA_MEMCPY2D_v2, CUmemorytype};

	let ctx = device as cudarc::driver::sys::CUcontext;
	unsafe { cuCtxSetCurrent(ctx) };

	let src_dev = (src as CUdeviceptr).wrapping_add(src_offset as usize);
	let dst_dev = (dst as CUdeviceptr).wrapping_add(dst_offset as usize);

	if src_pitch_bytes == dst_pitch_bytes && src_pitch_bytes == width_bytes {
		let total = (width_bytes as usize).saturating_mul(height as usize);
		let res = unsafe { cuMemcpyDtoD_v2(dst_dev, src_dev, total) };
		if res != CUresult::CUDA_SUCCESS {
			log::error!("[CUDA/buffer] cuMemcpyDtoD_v2 failed: {:?}", res);
			return Err("cuMemcpyDtoD_v2 failed");
		}
	} else {
		let cp = CUDA_MEMCPY2D_v2 {
			srcXInBytes: 0,
			srcY: 0,
			srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
			srcHost: std::ptr::null(),
			srcDevice: src_dev,
			srcArray: std::ptr::null_mut(),
			srcPitch: src_pitch_bytes as usize,
			dstXInBytes: 0,
			dstY: 0,
			dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
			dstHost: std::ptr::null_mut(),
			dstDevice: dst_dev,
			dstArray: std::ptr::null_mut(),
			dstPitch: dst_pitch_bytes as usize,
			WidthInBytes: width_bytes as usize,
			Height: height as usize,
		};
		let res = unsafe { cuMemcpy2D_v2(&cp) };
		if res != CUresult::CUDA_SUCCESS {
			log::error!("[CUDA/buffer] cuMemcpy2D_v2 failed: {:?}", res);
			return Err("cuMemcpy2D_v2 failed");
		}
	}

	Ok(())
}

/// # Safety: no GPU work may reference these buffers.
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
