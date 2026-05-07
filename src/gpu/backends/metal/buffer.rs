use std::sync::OnceLock;

use objc::{msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use crate::types::{compute_length_bytes, compute_row_bytes, mip_buffer_size_bytes, BufferKey, BufferObj, ImageBuffer};
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

pub(crate) unsafe fn allocate(device: *mut Object, length_bytes: u64, width: u32, height: u32, bpp: u32) -> *mut Object {
	const MAX_REASONABLE_BYTES: u64 = 512 * 1024 * 1024; // 512 MiB safety limit for image buffers
	if length_bytes > MAX_REASONABLE_BYTES {
		after_effects::log::error!(
			"[Metal] ABORT: refusing absurd buffer allocation of {} bytes ({} MiB) for {}x{} @ {} bpp — this is almost certainly a struct layout mismatch between Rust kernel_params! and the slang ConstantBuffer",
			length_bytes,
			length_bytes / 1024 / 1024,
			width,
			height,
			bpp
		);
		// Return a null buffer so the caller can detect failure instead of crashing the driver
		return std::ptr::null_mut();
	}
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
	unsafe { get_or_create_with_mips(device, width, height, bytes_per_pixel, 1, tag) }
}

/// Same as [`get_or_create`] but allocates a byte budget that fits a
/// `mip_levels`-deep mip chain. `mip_levels <= 1` behaves exactly like the
/// legacy form; higher values size the Metal buffer to
/// [`mip_buffer_size_bytes`] so the prgpu mip-downsample kernel can write
/// through without a re-alloc.
///
/// # Safety
/// See [`get_or_create`].
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
	let alloc_len = if mips <= 1 {
		compute_length_bytes(width, height, bytes_per_pixel)
	} else {
		mip_buffer_size_bytes(width, height, bytes_per_pixel, mips) as u64
	};
		let raw = match device {
			DeviceHandleInit::FromPtr(device) => {
				unsafe { allocate(device as *mut Object, alloc_len, width, height, bytes_per_pixel) as *mut std::ffi::c_void }
			}
		DeviceHandleInit::FromSuite((device_index, suite)) => {
			const MAX_REASONABLE_BYTES: u64 = 512 * 1024 * 1024;
			if alloc_len > MAX_REASONABLE_BYTES {
				after_effects::log::error!(
					"[Metal] ABORT (suite): refusing absurd buffer of {} bytes ({} MiB) for {}x{} @ {} bpp",
					alloc_len, alloc_len / 1024 / 1024, width, height, bytes_per_pixel
				);
				std::ptr::null_mut()
			} else {
				suite.allocate_device_memory(device_index, alloc_len as usize).unwrap_or_else(|e| {
					after_effects::log::error!("[Metal] GPUDevice suite allocation failed: {e:?}");
					std::ptr::null_mut()
				})
			}
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

/// Buffer-to-buffer GPU copy via a fresh `MTLBlitCommandEncoder`. Submits
/// its own `MTLCommandBuffer` to `command_queue` and waits on the GPU-
/// side fence before returning, so subsequent dispatches on the same
/// queue see the copied data.
///
/// Handles mismatched row pitches (src padded by Premiere vs. tight mip
/// buffer) via row-by-row blits. When both pitches match, a single flat
/// blit covers the whole region.
///
/// # Safety
/// - `command_queue`, `src`, `dst` must be a valid `MTLCommandQueue`,
///   `MTLBuffer`, `MTLBuffer` respectively (non-null).
/// - `src` must hold at least `src_pitch_bytes * height` bytes starting
///   at `src_offset`; same for `dst` with `dst_pitch_bytes`.
/// - No outstanding GPU work may read from `dst` concurrently.
pub unsafe fn copy_buffer(
	command_queue: *mut Object,
	src: *mut Object,
	src_offset: u64,
	src_pitch_bytes: u32,
	dst: *mut Object,
	dst_offset: u64,
	dst_pitch_bytes: u32,
	width_bytes: u32,
	height: u32,
) -> Result<(), &'static str> {
	if command_queue.is_null() || src.is_null() || dst.is_null() {
		return Err("copy_buffer: null handle");
	}

	let cmd: *mut Object = unsafe { msg_send![command_queue, commandBuffer] };
	if cmd.is_null() {
		return Err("copy_buffer: commandBuffer() returned null");
	}

	let enc: *mut Object = unsafe { msg_send![cmd, blitCommandEncoder] };
	if enc.is_null() {
		return Err("copy_buffer: blitCommandEncoder() returned null");
	}

	if src_pitch_bytes == dst_pitch_bytes && src_pitch_bytes == width_bytes {
		// Tight on both sides and matching pitch: one flat copy.
		let total = (width_bytes as u64) * (height as u64);
		unsafe {
			let _: () = msg_send![enc,
				copyFromBuffer: src sourceOffset: src_offset
				toBuffer: dst destinationOffset: dst_offset
				size: total as usize];
		}
	} else {
		// Different pitches: row-by-row flat copies.
		for y in 0..(height as u64) {
			let src_row_off = src_offset + y * (src_pitch_bytes as u64);
			let dst_row_off = dst_offset + y * (dst_pitch_bytes as u64);
			unsafe {
				let _: () = msg_send![enc,
					copyFromBuffer: src sourceOffset: src_row_off
					toBuffer: dst destinationOffset: dst_row_off
					size: width_bytes as usize];
			}
		}
	}

	unsafe {
		let _: () = msg_send![enc, endEncoding];
		let _: () = msg_send![cmd, commit];
		let _: () = msg_send![cmd, waitUntilCompleted];
	}
	Ok(())
}
