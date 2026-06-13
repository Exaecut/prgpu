use after_effects as ae;
use std::marker::PhantomData;

use crate::cpu::render::{CpuDispatchFn, CpuDispatchTileFn};
use crate::kernel::params::KernelParams;
use crate::types::Configuration;

/// Typed, dispatch-ready kernel descriptor produced by `kernel!`.
///
/// Holds every entry point the graph executor needs (shader bytes, entry
/// point name, CPU dispatch fns) so a render pass can be executed against
/// the active backend without per-effect wiring code.
pub struct Kernel<P: KernelParams> {
	name: &'static str,
	shader_src: &'static [u8],
	entry_point: &'static str,
	cpu_dispatch: CpuDispatchFn,
	cpu_dispatch_tile: CpuDispatchTileFn,
	_phantom: PhantomData<P>,
}

impl<P: KernelParams> Kernel<P> {
	pub const fn new(
		name: &'static str,
		shader_src: &'static [u8],
		entry_point: &'static str,
		cpu_dispatch: CpuDispatchFn,
		cpu_dispatch_tile: CpuDispatchTileFn,
	) -> Self {
		Self {
			name,
			shader_src,
			entry_point,
			cpu_dispatch,
			cpu_dispatch_tile,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub const fn name(&self) -> &'static str {
		self.name
	}

	#[inline]
	pub const fn shader_src(&self) -> &'static [u8] {
		self.shader_src
	}

	#[inline]
	pub const fn entry_point(&self) -> &'static str {
		self.entry_point
	}

	#[inline]
	pub const fn cpu_dispatch(&self) -> CpuDispatchFn {
		self.cpu_dispatch
	}

	#[inline]
	pub const fn cpu_dispatch_tile(&self) -> CpuDispatchTileFn {
		self.cpu_dispatch_tile
	}

	/// # Safety
	/// Caller upholds the prgpu `Configuration` buffer / pitch / lifetime contract:
	/// `dest_data` is non-null and writable, source pointers are valid for the
	/// dispatch, GPU device handles match the active context.
	#[inline]
	pub unsafe fn dispatch_gpu(&self, config: &Configuration, params: P) -> Result<(), &'static str> {
		unsafe {
			crate::gpu::backends::dispatch_kernel::<P>(config, params, self.shader_src, self.entry_point)
		}
	}

	#[inline]
	pub fn dispatch_cpu(
		&self,
		in_data: &ae::InData,
		in_layer: &ae::Layer,
		out_layer: &mut ae::Layer,
		config: &Configuration,
		params: P,
	) -> Result<(), ae::Error> {
		crate::cpu::render::render_cpu(
			self.name,
			in_data,
			in_layer,
			out_layer,
			config,
			self.cpu_dispatch,
			self.cpu_dispatch_tile,
			&params,
		)
	}

	/// AE-host-free CPU dispatch for resource→resource passes (mip chain
	/// downsample / upsample). Skips the `iterate_with` fast path; partitions
	/// the destination buffer directly via the rayon tile dispatcher.
	///
	/// # Safety
	/// `config.dest_data` must be non-null and back at least
	/// `dest_pitch_px * height * bytes_per_pixel` bytes; source pointers must
	/// be valid for the dispatch and follow the kernel's slot expectations.
	#[inline]
	pub unsafe fn dispatch_cpu_direct(&self, config: &Configuration, params: P) {
		unsafe {
			crate::cpu::render::render_cpu_direct(self.name, config, self.cpu_dispatch_tile, &params);
		}
	}
}
