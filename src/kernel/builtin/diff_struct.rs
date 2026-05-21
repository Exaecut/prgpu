//! Built-in pixel-difference kernel constant buffer.
//!
//! Used by `prgpu::testing::compare` to produce blackbody heatmaps for
//! reference / candidate render diffing. The shader matches `DiffParams` in
//! `prgpu/shaders/diff.slang` byte-for-byte.

use crate::kernel::params::KernelParams;

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

impl KernelParams for DiffParams {
	const SIZE: usize = core::mem::size_of::<Self>();
	const ALIGN: usize = core::mem::align_of::<Self>();
}
