use cudarc::driver::sys::{
    CUcontext, CUdeviceptr, CUresult, cuCtxSetCurrent, cuMemAlloc_v2, cuMemFree_v2,
};
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::{collections::HashMap, ffi::c_void};

use after_effects::log;
use crate::DeviceHandleInit;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferKey {
    pub device: usize,
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub tag: u32,
}

/// Wraps a CUdeviceptr (device memory address stored as u64).
/// The value is the device pointer itself, NOT a host pointer to a device pointer.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BufferObj {
    pub raw: *mut c_void,
}

unsafe impl Send for BufferObj {}
unsafe impl Sync for BufferObj {}

#[derive(Clone, Copy)]
pub struct ImageBuffer {
    pub buf: BufferObj,
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub row_bytes: u32,
    pub pitch_px: u32,
}

static CACHE: OnceLock<Mutex<HashMap<BufferKey, BufferObj>>> = OnceLock::new();

#[inline]
fn compute_row_bytes(width: u32, bytes_per_pixel: u32) -> u32 {
    width.saturating_mul(bytes_per_pixel)
}

#[inline]
fn compute_length_bytes(width: u32, height: u32, bytes_per_pixel: u32) -> u64 {
    (width as u64) * (height as u64) * (bytes_per_pixel as u64)
}

/// Allocates device memory via cuMemAlloc_v2. Returns a device pointer as `*mut c_void`.
///
/// # Safety
/// - `device` must be a valid CUcontext pointer.
/// - Caller owns the returned allocation and must free with `cuMemFree_v2`.
pub unsafe fn create_raw_buffer(device: *mut c_void, length_bytes: u64) -> *mut c_void {
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

/// Allocates an image-sized device buffer (width * height * bpp).
///
/// # Safety
/// - `device` must be a valid CUcontext pointer.
pub unsafe fn create_texture_buffer(
    device: *mut c_void,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> *mut c_void {
    let length = compute_length_bytes(width, height, bytes_per_pixel);
    unsafe { create_raw_buffer(device, length) }
}

/// Gets a cached buffer or creates one. Returns an `ImageBuffer` view.
///
/// # Safety
/// - `device` must be a valid CUcontext pointer (FromPtr) or valid suite handle (FromSuite).
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
                let raw =
                    unsafe { create_texture_buffer(device, width, height, bytes_per_pixel) };
                if raw.is_null() {
                    log::error!("[CUDA] buffer allocation failed for {}x{} bpp={}", width, height, bytes_per_pixel);
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

/// Frees all cached device buffers.
///
/// # Safety
/// No GPU work may reference these buffers. No concurrent access to the cache.
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
