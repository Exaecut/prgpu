pub mod diff;
pub mod helpers;
pub mod mip;

/// Declare a GPU kernel (CUDA + Metal) and its CPU fallback dispatch.
///
/// `declare_kernel!(vignette, VignetteParams);` emits a per-kernel module
/// (`vignette::*`) plus deprecated top-level aliases for backward compat:
///
/// ```ignore
/// // New API (Phase 2 onward):
/// vignette::SHADER_SRC          // shader bytes for the active backend
/// vignette::ENTRY_POINT         // entry-point name
/// vignette::gpu(cfg, params)    // GPU dispatch
/// vignette::cpu(...)            // CPU dispatch wrapper around render_cpu
/// vignette::CPU_DISPATCH        // raw C ABI fn pointer (per-pixel)
/// vignette::CPU_DISPATCH_TILE   // raw C ABI fn pointer (tile)
/// vignette::kernel()            // returns Kernel<VignetteParams>
///
/// // Legacy (deprecated, will be removed):
/// vignette(cfg, params)
/// vignette_cpu(...)
/// VIGNETTE_CPU_DISPATCH / VIGNETTE_CPU_DISPATCH_TILE
/// ```
///
/// The generated `kernel()` constructor returns a [`crate::Kernel<P>`] that
/// the graph executor uses to route per-pass dispatch by backend.
#[macro_export]
macro_rules! declare_kernel {
	($name:ident, $user_params_ty:ty) => {
		pub mod $name {
			#[allow(unused_imports)]
			use super::*;

			pub const SHADER_SRC: &[u8] = {
				#[cfg(gpu_backend = "metal")]
				{ $crate::include_shader!($name, metal) }

				#[cfg(gpu_backend = "cuda")]
				{ $crate::include_shader!($name, cuda) }

				#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
				{ &[] }
			};

			pub const ENTRY_POINT: &str = stringify!($name);

			$crate::paste::paste! {
				unsafe extern "C" {
					pub fn [<$name _cpu_dispatch>](
						gid_x: u32,
						gid_y: u32,
						buffers: *const *const std::ffi::c_void,
						transition_params: *const std::ffi::c_void,
						user_params: *const std::ffi::c_void,
					);

					/// Tile entry: loops `y ∈ [y0, y1) × x ∈ [0, width)` in C.
					pub fn [<$name _cpu_dispatch_tile>](
						y0: u32,
						y1: u32,
						width: u32,
						buffers: *const *const std::ffi::c_void,
						transition_params: *const std::ffi::c_void,
						user_params: *const std::ffi::c_void,
					);
				}

				pub const CPU_DISPATCH: $crate::cpu::render::CpuDispatchFn = [<$name _cpu_dispatch>];
				pub const CPU_DISPATCH_TILE: $crate::cpu::render::CpuDispatchTileFn = [<$name _cpu_dispatch_tile>];
			}

			/// # Safety
			/// `config` must satisfy the prgpu `Configuration` buffer/pitch contract
			/// (non-null `dest_data`, valid GPU handles, dimensions consistent with
			/// the bound buffers). Caller is responsible for synchronisation.
			pub unsafe fn gpu(
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), &'static str> {
				$crate::backends::dispatch_kernel::<$user_params_ty>(config, user_params, SHADER_SRC, ENTRY_POINT)
			}

			pub fn cpu(
				in_data: &after_effects::InData,
				in_layer: &after_effects::Layer,
				out_layer: &mut after_effects::Layer,
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), after_effects::Error> {
				$crate::paste::paste! {
					$crate::cpu::render::render_cpu(
						stringify!($name),
						in_data,
						in_layer,
						out_layer,
						config,
						[<$name _cpu_dispatch>],
						[<$name _cpu_dispatch_tile>],
						&user_params,
					)
				}
			}

			pub fn kernel() -> $crate::Kernel<$user_params_ty> {
				$crate::Kernel::new(
					stringify!($name),
					SHADER_SRC,
					ENTRY_POINT,
					CPU_DISPATCH,
					CPU_DISPATCH_TILE,
					gpu,
					cpu,
				)
			}
		}

		$crate::paste::paste! {
			#[allow(non_upper_case_globals)]
			pub const [<$name:upper _CPU_DISPATCH>]: $crate::cpu::render::CpuDispatchFn = $name::CPU_DISPATCH;

			#[allow(non_upper_case_globals)]
			pub const [<$name:upper _CPU_DISPATCH_TILE>]: $crate::cpu::render::CpuDispatchTileFn = $name::CPU_DISPATCH_TILE;
		}

		#[deprecated(note = "use `<name>::gpu` or `<name>::kernel()` for graph execution")]
		#[allow(dead_code)]
		pub unsafe fn $name(
			config: &$crate::types::Configuration,
			user_params: $user_params_ty,
		) -> Result<(), &'static str> {
			unsafe { $name::gpu(config, user_params) }
		}

		$crate::paste::paste! {
			#[deprecated(note = "use `<name>::cpu` or `<name>::kernel()` for graph execution")]
			#[allow(dead_code)]
			pub fn [<$name _cpu>](
				in_data: &after_effects::InData,
				in_layer: &after_effects::Layer,
				out_layer: &mut after_effects::Layer,
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), after_effects::Error> {
				$name::cpu(in_data, in_layer, out_layer, config, user_params)
			}
		}
	};
}
