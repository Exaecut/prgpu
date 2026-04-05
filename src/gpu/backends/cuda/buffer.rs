use cudarc::driver::sys::{
    CUcontext, CUdeviceptr, CUresult, cuCtxSetCurrent, cuMemAlloc_v2, cuMemFree_v2,
};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::{collections::HashMap, ffi::c_void};

use after_effects::log;
use crate::types::{BufferKey, BufferObj, ImageBuffer, compute_row_bytes, compute_length_bytes};
use crate::DeviceHandleInit;

static CACHE: OnceLock<Mutex<HashMap<BufferKey, BufferObj>>> = OnceLock::new();

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
            log::error!(
                "[CUDA] cuMemAlloc_v2 failed: {:?} (requested {} bytes)",
                err,
                length_bytes
            );
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `device` must be a valid CUcontext pointer (FromPtr) or valid suite handle (FromSuite).
pub unsafe fn get_or_create(
    device: DeviceHandleInit,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    tag: u32,
) -> ImageBuffer {
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
                let raw = unsafe { allocate(device, length) };
                if raw.is_null() {
                    log::error!(
                        "[CUDA] buffer allocation failed for {}x{} bpp={}",
                        width, height, bytes_per_pixel
                    );
                }
                let obj = BufferObj { raw };
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
            let allocated = suite
                .allocate_device_memory(device_index, length)
                .unwrap_or_else(|e| {
                    log::error!("[CUDA] GPUDevice suite allocation failed: {e:?}");
                    std::ptr::null_mut()
                });

            let device_handle = suite
                .device_info(device_index)
                .map(|info| info.outDeviceHandle as usize)
                .unwrap_or(0);

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
        for (_key, buf) in guard.drain() {
            if !buf.raw.is_null() {
                let devptr = buf.raw as CUdeviceptr;
                let res = unsafe { cuMemFree_v2(devptr) };
                if res != CUresult::CUDA_SUCCESS {
                    log::error!("[CUDA] cuMemFree_v2 failed: {:?}", res);
                }
            }
        }
    }
}
