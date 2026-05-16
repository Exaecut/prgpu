//! Built-in pixel-difference kernel. Generates a blackbody heatmap.

use crate::declare_kernel;

/// Must match `DiffParams` in `diff.slang` byte-for-byte.
/// `_pad*` fills to 32 bytes (8 × u32) for vec4 alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DiffParams {
    pub tol_r: f32,
    pub tol_g: f32,
    pub tol_b: f32,
    pub tol_a: f32,
    pub smooth_a: f32,
    pub smooth_b: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

declare_kernel!(diff, DiffParams);
