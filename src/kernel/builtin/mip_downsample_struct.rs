//! Built-in mip-chain downsampler constant buffer.
//!
//! `_pad*` aligns the slang ConstantBuffer to a 16-byte vec4 boundary,
//! matching the `uint _pad0; uint _pad1; uint _pad2;` fields in
//! `prgpu/shaders/mip_downsample.slang`.

use crate::kernel::params::KernelParams;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MipDownsampleParams {
	pub src_lod: u32,
	pub _pad0: u32,
	pub _pad1: u32,
	pub _pad2: u32,
}

impl KernelParams for MipDownsampleParams {
	const SIZE: usize = core::mem::size_of::<Self>();
	const ALIGN: usize = core::mem::align_of::<Self>();
}
