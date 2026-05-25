use cudarc::driver::sys::{cuCtxSetCurrent, cuMemAlloc_v2, cuMemFree_v2, CUcontext, CUdeviceptr, CUresult};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::ffi::c_void;

use crate::types::{compute_length_bytes, compute_row_bytes, mip_buffer_size_bytes, BufferKey, BufferObj, ImageBuffer};
use crate::{Configuration, DeviceHandleInit};
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

/// Cache-aware variant: returns `(buffer, was_hit)`. Callers that need to
/// populate the buffer only on first allocation (e.g. source snapshot) use
/// `was_hit` to skip the upload on cache hit. See `prepare_source_snapshot`.
///
/// # Safety: see `get_or_create`.
pub unsafe fn get_or_create_returning_hit(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> (ImageBuffer, bool) {
	unsafe { get_or_create_with_mips_inner(device, width, height, bytes_per_pixel, 1, tag) }
}

/// Like `get_or_create` but sized for an `mip_levels`-deep mip chain.
///
/// # Safety: see `get_or_create`.
pub unsafe fn get_or_create_with_mips(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, mip_levels: u32, tag: u32) -> ImageBuffer {
	unsafe { get_or_create_with_mips_inner(device, width, height, bytes_per_pixel, mip_levels, tag) }.0
}

unsafe fn get_or_create_with_mips_inner(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, mip_levels: u32, tag: u32) -> (ImageBuffer, bool) {
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
		let ptr = existing.raw;
		drop(guard);
		return (
			ImageBuffer {
				buf: BufferObj { raw: ptr },
				width,
				height,
				bytes_per_pixel,
				row_bytes: compute_row_bytes(width, bytes_per_pixel),
				pitch_px: width,
			},
			true,
		);
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
}

/// Buffer-to-buffer device copy. `cuMemcpy2D_v2` for mismatched pitches
/// (Premiere's padded source vs. tight mip buffer); falls back to flat
/// `cuMemcpyDtoD_v2` when pitches match.
///
/// Synchronous on the default stream so subsequent dispatches see the copied data.
///
/// Signature mirrors the Metal backend so callers stay backend-agnostic; both
/// backends pull whichever Configuration field they need (Metal: command queue
/// from `command_queue_handle`; CUDA: CUcontext from `context_handle`).
///
/// On CUDA Premiere `device_handle` is a CUdevice ordinal — NOT a CUcontext;
/// `cuCtxSetCurrent(device_handle)` returns `CUDA_ERROR_INVALID_CONTEXT` and
/// the subsequent memcpy fails. Always pull the context from `context_handle`.
///
/// # Safety
/// - `config.context_handle` must hold the CUcontext that owns both `src` and `dst`.
/// - Both must hold at least `pitch_bytes * height` bytes from their offsets.
/// - No other GPU work may touch `dst` concurrently.
pub unsafe fn copy_buffer(
	config: &Configuration,
	src: *mut c_void,
	src_offset: u64,
	src_pitch_bytes: u32,
	dst: *mut c_void,
	dst_offset: u64,
	dst_pitch_bytes: u32,
	width_bytes: u32,
	height: u32,
) -> Result<(), &'static str> {
	use cudarc::driver::sys::{cuMemcpy2D_v2, CUDA_MEMCPY2D_v2, CUmemorytype};

	let Some(ctx_ptr) = config.context_handle else {
		log::error!("[CUDA/buffer] copy_buffer: config.context_handle is None");
		return Err("copy_buffer: missing CUcontext");
	};
	if ctx_ptr.is_null() {
		log::error!("[CUDA/buffer] copy_buffer: config.context_handle is null");
		return Err("copy_buffer: null CUcontext");
	}
	let ctx = ctx_ptr as CUcontext;
	let set = unsafe { cuCtxSetCurrent(ctx) };
	if set != CUresult::CUDA_SUCCESS {
		log::error!("[CUDA/buffer] copy_buffer: cuCtxSetCurrent failed: {:?}", set);
		return Err("copy_buffer: cuCtxSetCurrent failed");
	}

	let src_dev = (src as CUdeviceptr).wrapping_add(src_offset);
	let dst_dev = (dst as CUdeviceptr).wrapping_add(dst_offset);

	// Always go through the 2D copy with `CU_MEMORYTYPE_UNIFIED` so CUDA can
	// auto-detect the actual memory type via UVA. The Premiere RE shows source
	// PPix may be `cuMemHostRegister`-wrapped pages or `cuMemHostAlloc`-pinned
	// memory (visible as `HostMemory` pool in `<GF.CUDAError>` JSON). Declaring
	// `srcMemoryType = CU_MEMORYTYPE_DEVICE` against a host-origin UVA pointer
	// makes CUDA reject with `CUDA_ERROR_INVALID_VALUE`. UNIFIED works for both
	// pure-device and host-UVA pointers, and the prior `cuMemcpyDtoD_v2`
	// fast-path inherits the same constraint, so we drop it.
	let cp = CUDA_MEMCPY2D_v2 {
		srcXInBytes: 0,
		srcY: 0,
		srcMemoryType: CUmemorytype::CU_MEMORYTYPE_UNIFIED,
		srcHost: std::ptr::null(),
		srcDevice: src_dev,
		srcArray: std::ptr::null_mut(),
		srcPitch: src_pitch_bytes as usize,
		dstXInBytes: 0,
		dstY: 0,
		dstMemoryType: CUmemorytype::CU_MEMORYTYPE_UNIFIED,
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
