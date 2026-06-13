use std::sync::OnceLock;

use objc::{msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use crate::types::{compute_length_bytes, compute_row_bytes, mip_buffer_size_bytes, BufferKey, BufferObj, ImageBuffer};
use crate::types::{Configuration, DeviceHandleInit};

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

	/// Insert, evicting LRU when at capacity. Returns the evicted `BufferObj` (caller releases it).
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
		// Null buffer lets the caller fail gracefully instead of crashing the driver.
		return std::ptr::null_mut();
	}
	let opts = StorageMode::Private.as_resource_options();
	msg_send![device, newBufferWithLength: length_bytes options: opts]
}

unsafe fn free_buffer(buf: BufferObj) {
	if !buf.raw.is_null() {
		let _: () = msg_send![buf.raw as *mut Object, release];
	}
}

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

/// Like `get_or_create` but sized for an `mip_levels`-deep mip chain via `mip_buffer_size_bytes`.
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
		return (
			ImageBuffer {
				buf: existing,
				width,
				height,
				bytes_per_pixel,
				row_bytes: compute_row_bytes(width, bytes_per_pixel),
				pitch_px: width,
			},
			true,
		);
	}

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

/// Buffer-to-buffer GPU copy via an `MTLBlitCommandEncoder`. Inside a frame
/// scope the blit encodes into the frame command buffer (ordered with the
/// surrounding passes, no stall); otherwise it submits its own command buffer
/// and waits before returning.
///
/// Row-by-row blits when pitches mismatch; one flat blit when they match.
///
/// Signature mirrors the CUDA backend so callers stay backend-agnostic; both
/// backends pull whichever Configuration field they need (Metal: command queue,
/// CUDA: CUcontext).
///
/// # Safety
/// - `config.command_queue_handle`, `src`, `dst` must be valid non-null Metal handles.
/// - Both must hold at least `pitch_bytes * height` bytes from their offsets.
/// - No outstanding GPU work may read from `dst` concurrently.
pub unsafe fn copy_buffer(
	config: &Configuration,
	src: *mut std::ffi::c_void,
	src_offset: u64,
	src_pitch_bytes: u32,
	dst: *mut std::ffi::c_void,
	dst_offset: u64,
	dst_pitch_bytes: u32,
	width_bytes: u32,
	height: u32,
) -> Result<(), &'static str> {
	let command_queue = config.command_queue_handle as *mut Object;
	let src = src as *mut Object;
	let dst = dst as *mut Object;

	if command_queue.is_null() || src.is_null() || dst.is_null() {
		return Err("copy_buffer: null handle");
	}

	let in_frame_scope = super::frame_scope::is_active();
	let cmd: *mut Object = if in_frame_scope {
		super::frame_scope::command_buffer()
	} else {
		unsafe { msg_send![command_queue, commandBuffer] }
	};
	if cmd.is_null() {
		return Err("copy_buffer: commandBuffer() returned null");
	}

	let enc: *mut Object = unsafe { msg_send![cmd, blitCommandEncoder] };
	if enc.is_null() {
		return Err("copy_buffer: blitCommandEncoder() returned null");
	}

	if src_pitch_bytes == dst_pitch_bytes && src_pitch_bytes == width_bytes {
		// Tight on both sides + matching pitch: one flat copy.
		let total = (width_bytes as u64) * (height as u64);
		unsafe {
			let _: () = msg_send![enc,
				copyFromBuffer: src sourceOffset: src_offset
				toBuffer: dst destinationOffset: dst_offset
				size: total as usize];
		}
	} else {
		// Mismatched pitches: row-by-row copies.
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
	}
	if !in_frame_scope {
		unsafe {
			let _: () = msg_send![cmd, commit];
			let _: () = msg_send![cmd, waitUntilCompleted];
		}
	}
	Ok(())
}
