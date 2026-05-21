use after_effects as ae;

use crate::cpu::render::{CpuDispatchFn, CpuDispatchTileFn};
use crate::types::{Configuration, KernelParams};

/// GPU dispatch entry: hands `(config, user_params)` to the active backend
/// (`backends::dispatch_kernel`) using the shader bytes the kernel was built with.
pub type GpuDispatchFn<P> = unsafe fn(&Configuration, P) -> Result<(), &'static str>;

/// CPU render entry: writes the destination AE layer using the static C++
/// dispatch fns Slang produced for this kernel.
pub type CpuRenderFn<P> = fn(&ae::InData, &ae::Layer, &mut ae::Layer, &Configuration, P) -> Result<(), ae::Error>;

/// Typed, dispatch-ready kernel descriptor produced by `declare_kernel!`.
///
/// Holds every entry point the graph executor needs (`shader_src`, entry
/// point name, CPU dispatch fns, GPU/CPU adapters) so a render pass can be
/// executed against the active backend without per-effect wiring code.
pub struct Kernel<P: KernelParams> {
	name: &'static str,
	shader_src: &'static [u8],
	entry_point: &'static str,
	cpu_dispatch: CpuDispatchFn,
	cpu_dispatch_tile: CpuDispatchTileFn,
	gpu_dispatch: GpuDispatchFn<P>,
	cpu_render: CpuRenderFn<P>,
}

impl<P: KernelParams> Kernel<P> {
	pub const fn new(
		name: &'static str,
		shader_src: &'static [u8],
		entry_point: &'static str,
		cpu_dispatch: CpuDispatchFn,
		cpu_dispatch_tile: CpuDispatchTileFn,
		gpu_dispatch: GpuDispatchFn<P>,
		cpu_render: CpuRenderFn<P>,
	) -> Self {
		Self {
			name,
			shader_src,
			entry_point,
			cpu_dispatch,
			cpu_dispatch_tile,
			gpu_dispatch,
			cpu_render,
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
		unsafe { (self.gpu_dispatch)(config, params) }
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
		(self.cpu_render)(in_data, in_layer, out_layer, config, params)
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
