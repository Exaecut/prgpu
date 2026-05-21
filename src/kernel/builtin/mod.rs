//! Built-in GPU kernels that ship with prgpu.
//!
//! Each kernel splits into two pieces:
//! 1. `<name>_struct.rs` defines the `<Name>Params` constant-buffer struct
//!    + `KernelParams` impl. Private to this module.
//! 2. `mod.rs` (this file) re-exports the struct and invokes
//!    `declare_kernel!(<name>, <Name>Params)` directly. The macro emits a
//!    `pub mod <name>` containing every dispatch entry point — landing the
//!    public path at `prgpu::kernel::builtin::<name>::*` instead of the
//!    redundant `kernels::<name>::<name>::*` shape the old layout produced.

use crate::declare_kernel;

mod diff_struct;
pub use diff_struct::DiffParams;

declare_kernel!(diff, DiffParams);

mod mip_downsample_struct;
pub use mip_downsample_struct::MipDownsampleParams;

declare_kernel!(mip_downsample, MipDownsampleParams);
