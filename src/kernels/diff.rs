//! Built-in pixel-difference kernel. Generates a colour-coded heatmap.

use crate::declare_kernel;

/// Tolerance per channel, in [0, 1]. Must match `DiffParams` in `diff.slang`
/// byte-for-byte, including the three `_pad` fields for vec4 alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DiffParams {
    pub tol_r: f32,
    pub tol_g: f32,
    pub tol_b: f32,
    pub tol_a: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

declare_kernel!(diff, DiffParams);
