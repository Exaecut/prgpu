//! Built-in GPU kernels that ship with prgpu.
//!
//! Each kernel splits into two pieces:
//! 1. `<name>_struct.rs` defines the `<Name>Params` constant-buffer struct
//!    + `KernelParams` impl. Private to this module.
//! 2. `mod.rs` (this file) re-exports the struct and wires the dispatch
//!    module with `__kernel_dispatch_externs!`.

mod diff_struct;
pub use diff_struct::DiffParams;

prgpu::paste::paste! {
	unsafe extern "C" {
		pub fn [<diff _cpu_dispatch>](
			gid_x: u32,
			gid_y: u32,
			buffers: *const *const ::core::ffi::c_void,
			transition_params: *const ::core::ffi::c_void,
			user_params: *const ::core::ffi::c_void,
		);

		pub fn [<diff _cpu_dispatch_tile>](
			y0: u32,
			y1: u32,
			width: u32,
			buffers: *const *const ::core::ffi::c_void,
			transition_params: *const ::core::ffi::c_void,
			user_params: *const ::core::ffi::c_void,
		);
	}
}

pub mod diff {
	pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/diff.shader"));

	pub const ENTRY_POINT: &str = "diff";

	pub fn kernel() -> crate::Kernel<super::DiffParams> {
		crate::Kernel::new("diff", SHADER, "diff", super::diff_cpu_dispatch, super::diff_cpu_dispatch_tile)
	}
}

mod mip_downsample_struct;
pub use mip_downsample_struct::MipDownsampleParams;

prgpu::paste::paste! {
	unsafe extern "C" {
		pub fn [<mip_downsample _cpu_dispatch>](
			gid_x: u32,
			gid_y: u32,
			buffers: *const *const ::core::ffi::c_void,
			transition_params: *const ::core::ffi::c_void,
			user_params: *const ::core::ffi::c_void,
		);

		pub fn [<mip_downsample _cpu_dispatch_tile>](
			y0: u32,
			y1: u32,
			width: u32,
			buffers: *const *const ::core::ffi::c_void,
			transition_params: *const ::core::ffi::c_void,
			user_params: *const ::core::ffi::c_void,
		);
	}
}

pub mod mip_downsample {
	pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mip_downsample.shader"));

	pub const ENTRY_POINT: &str = "mip_downsample";

	pub fn kernel() -> crate::Kernel<super::MipDownsampleParams> {
		crate::Kernel::new(
			"mip_downsample",
			SHADER,
			"mip_downsample",
			super::mip_downsample_cpu_dispatch,
			super::mip_downsample_cpu_dispatch_tile,
		)
	}
}
