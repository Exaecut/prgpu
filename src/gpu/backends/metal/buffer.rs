use std::collections::HashMap;
use std::sync::OnceLock;

use objc::{msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use crate::DeviceHandleInit;

/// Key that uniquely identifies a cached GPU buffer allocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferKey {
    /// Device pointer (MTLDevice*) used for allocation - cast to usize for hashing
    pub device: usize,
    /// Width in pixels (for convenience when used as an image buffer)
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Bytes per pixel (e.g. 16 for float4, 8 for half4)
    pub bytes_per_pixel: u32,
    /// Optional tag to differentiate multiple buffers of the same size (0 by default)
    pub tag: u32,
}

/// Thin wrapper around an MTLBuffer* that we explicitly mark Send + Sync.
/// You are responsible for lifetime via `cleanup()`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BufferObj {
    pub raw: *mut Object,
}

// We commonly pass MTLBuffer* across threads in render pipelines.
// Marking these wrappers as Send/Sync is a deliberate design choice here.
unsafe impl Send for BufferObj {}
unsafe impl Sync for BufferObj {}

/// Public view returned to callers. Copying is cheap and does not affect ownership.
/// The underlying buffer is owned by the cache and freed by `cleanup()`.
#[derive(Clone, Copy)]
pub struct ImageBuffer {
    pub buf: BufferObj,
    pub width: u32,
    pub height: u32,
    /// Bytes per pixel
    pub bytes_per_pixel: u32,
    /// Row bytes in bytes (for APIs that want bytes)
    pub row_bytes: u32,
    /// Pitch in pixels (what your shaders use)
    pub pitch_px: u32,
}

// Internal cache: one buffer per (device, size, bpp, tag).
static CACHE: OnceLock<Mutex<HashMap<BufferKey, BufferObj>>> = OnceLock::new();

#[inline]
fn compute_row_bytes(width: u32, bytes_per_pixel: u32) -> u32 {
    width.saturating_mul(bytes_per_pixel)
}

#[inline]
fn compute_length_bytes(width: u32, height: u32, bytes_per_pixel: u32) -> u64 {
    (width as u64) * (height as u64) * (bytes_per_pixel as u64)
}

/// Low-level creator: returns a new MTLBuffer* with given length.
/// Storage mode: Shared (options = 0). Adjust if you later need Private.
///
/// # Safety
/// - `device` must be a valid pointer to an MTLDevice*.
/// - The caller must ensure that the returned buffer is properly managed and released when no longer needed.
pub unsafe fn create_raw_buffer(device: *mut Object, length_bytes: u64) -> *mut Object {
    let buf: *mut Object = msg_send![device, newBufferWithLength: length_bytes options: 0u64];
    buf
}

/// Create an "image-like" buffer sized width*height with the given bytes_per_pixel.
///
/// # Safety
/// - `device` must be a valid pointer to an MTLDevice*.
/// - The caller must ensure that the returned buffer is properly managed and released when no longer needed.
pub unsafe fn create_texture_buffer(
    device: *mut Object,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> *mut Object {
    let length = compute_length_bytes(width, height, bytes_per_pixel);
    unsafe { create_raw_buffer(device, length) }
}

/// Get a cached buffer or create-and-cache one if absent.
/// Returns an `ImageBuffer` view with useful stride info populated.
///
/// # Safety
/// - `device` must be a valid pointer to an MTLDevice*.
/// - The caller must ensure that the returned buffer is properly managed and released when no longer needed.
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
                let raw = unsafe {
                    create_texture_buffer(device as *mut Object, width, height, bytes_per_pixel)
                };
                let obj = BufferObj { raw };
                guard.insert(key, obj);
                obj
            };

            let row_bytes = compute_row_bytes(width, bytes_per_pixel);
            let pitch_px = width;

            ImageBuffer {
                buf,
                width,
                height,
                bytes_per_pixel,
                row_bytes,
                pitch_px,
            }
        }
        DeviceHandleInit::FromSuite((device_index, suite)) => {
            if let Ok(allocated) = suite.allocate_device_memory(
                device_index,
                compute_length_bytes(width, height, bytes_per_pixel) as usize,
            ) {
                let key = BufferKey {
                    device: suite.device_info(device_index).unwrap().outDeviceHandle as usize,
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
                    let obj = BufferObj {
                        raw: allocated as *mut Object,
                    };
                    guard.insert(key, obj);
                    obj
                };

                let row_bytes = compute_row_bytes(width, bytes_per_pixel);
                let pitch_px = width;

                ImageBuffer {
                    buf,
                    width,
                    height,
                    bytes_per_pixel,
                    row_bytes,
                    pitch_px,
                }
            } else {
                panic!("Failed to allocate device memory via GPUDevice suite");
            }
        }
    }
}

/// Release all cached buffers. Call this from your plugin's global_destroy.
///
/// # Safety
/// - This function must only be called when you are certain that no GPU work still references these buffers.
/// - Ensure that no other threads are accessing the cached buffers while this function is being executed.
pub unsafe fn cleanup() {
    if let Some(map) = CACHE.get() {
        let mut guard = map.lock();
        for (_, b) in guard.drain() {
            if !b.raw.is_null() {
                let _: () = msg_send![b.raw, release];
            }
        }
    }
}
