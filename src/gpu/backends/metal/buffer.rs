use std::collections::HashMap;
use std::sync::OnceLock;

use objc::{msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use crate::types::{compute_length_bytes, compute_row_bytes, BufferKey, BufferObj, ImageBuffer};
use crate::DeviceHandleInit;

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

static CACHE: OnceLock<Mutex<HashMap<BufferKey, BufferObj>>> = OnceLock::new();

/// # Safety
/// `device` must be a valid MTLDevice pointer.
pub(crate) unsafe fn allocate(device: *mut Object, length_bytes: u64) -> *mut Object {
	let opts = StorageMode::Private.as_resource_options();
	msg_send![device, newBufferWithLength: length_bytes options: opts]
}

/// # Safety
/// `device` must be a valid MTLDevice pointer (FromPtr) or valid suite handle (FromSuite).
pub unsafe fn get_or_create(device: DeviceHandleInit, width: u32, height: u32, bytes_per_pixel: u32, tag: u32) -> ImageBuffer {
	match device {
		DeviceHandleInit::FromPtr(device) => {
			let key = BufferKey {
				device: device as usize,
				width,
				height,
				bytes_per_pixel,
				tag,
			};

			let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
			let mut guard = map.lock();

			let buf = if let Some(existing) = guard.get(&key) {
				*existing
			} else {
				let length = compute_length_bytes(width, height, bytes_per_pixel);
				let raw = unsafe { allocate(device as *mut Object, length) };
				let obj = BufferObj {
					raw: raw as *mut std::ffi::c_void,
				};
				guard.insert(key, obj);
				obj
			};

			ImageBuffer {
				buf,
				width,
				height,
				bytes_per_pixel,
				row_bytes: compute_row_bytes(width, bytes_per_pixel),
				pitch_px: width,
			}
		}
		DeviceHandleInit::FromSuite((device_index, suite)) => {
			let length = compute_length_bytes(width, height, bytes_per_pixel) as usize;
			let allocated = suite.allocate_device_memory(device_index, length).unwrap_or_else(|e| {
				after_effects::log::error!("[Metal] GPUDevice suite allocation failed: {e:?}");
				std::ptr::null_mut()
			});

			let device_handle = suite.device_info(device_index).map(|info| info.outDeviceHandle as usize).unwrap_or(0);

			let key = BufferKey {
				device: device_handle,
				width,
				height,
				bytes_per_pixel,
				tag,
			};

			let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
			let mut guard = map.lock();

			let buf = if let Some(existing) = guard.get(&key) {
				*existing
			} else {
				let obj = BufferObj { raw: allocated };
				guard.insert(key, obj);
				obj
			};

			ImageBuffer {
				buf,
				width,
				height,
				bytes_per_pixel,
				row_bytes: compute_row_bytes(width, bytes_per_pixel),
				pitch_px: width,
			}
		}
	}
}

/// # Safety
/// No GPU work may reference these buffers.
pub unsafe fn cleanup() {
	if let Some(map) = CACHE.get() {
		let mut guard = map.lock();
		for (_, b) in guard.drain() {
			if !b.raw.is_null() {
				let _: () = msg_send![b.raw as *mut Object, release];
			}
		}
	}
}
