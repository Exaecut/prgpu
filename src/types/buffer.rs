use std::ffi::c_void;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferKey {
    pub device: usize,
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub tag: u32,
    /// Mip levels the buffer was allocated for. `1` = no mip chain. Keyed so
    /// the same dims requested with and without a mip chain resolve to separate
    /// cache slots instead of sharing a too-small allocation.
    pub mip_levels: u32,
}

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

#[inline]
pub fn compute_row_bytes(width: u32, bytes_per_pixel: u32) -> u32 {
    width.saturating_mul(bytes_per_pixel)
}

#[inline]
pub fn compute_length_bytes(width: u32, height: u32, bytes_per_pixel: u32) -> u64 {
    (width as u64) * (height as u64) * (bytes_per_pixel as u64)
}
