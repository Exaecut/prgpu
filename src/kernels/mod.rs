pub mod helpers;
pub mod mip;

/// Declare a GPU kernel (CUDA + Metal) and its CPU fallback dispatch.
///
/// `declare_kernel!(vignette, VignetteParams);` generates:
/// - `VIGNETTE_SHADER_SRC` (`.metallib` bytes on Metal, `.ptx` bytes on CUDA)
/// - `VIGNETTE_KERNEL_ENTRY_POINT: &'static str` (`stringify!($name)` — required
///   to be `'static` because the CUDA pipeline cache stores it as a key)
/// - `vignette(config, user_params)` (GPU dispatch)
/// - `vignette_cpu(in_data, in_layer, out_layer, config, user_params)` (CPU dispatch)
#[macro_export]
macro_rules! declare_kernel {
	($name:ident, $user_params_ty:ty) => {
		$crate::paste::paste! {

			#[allow(non_upper_case_globals)]
			const [<$name:upper _SHADER_SRC>]: &[u8] = {
				#[cfg(gpu_backend = "metal")]
				{ $crate::include_shader!($name, metal) }

				#[cfg(gpu_backend = "cuda")]
				{ $crate::include_shader!($name, cuda) }

				#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
				{ &[] }
			};
		}

		$crate::paste::paste! {
			#[allow(non_upper_case_globals)]
			const [<$name:upper _KERNEL_ENTRY_POINT>]: &str = stringify!($name);
		}

		$crate::paste::paste! {
			pub unsafe fn $name(
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), &'static str> {
				$crate::backends::dispatch_kernel::<$user_params_ty>(
					config,
					user_params,
					[<$name:upper _SHADER_SRC>],
					[<$name:upper _KERNEL_ENTRY_POINT>],
				)
			}
		}

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

			#[allow(non_upper_case_globals)]
			pub const [<$name:upper _CPU_DISPATCH>]: $crate::cpu::render::CpuDispatchFn =
				[<$name _cpu_dispatch>];

			#[allow(non_upper_case_globals)]
			pub const [<$name:upper _CPU_DISPATCH_TILE>]: $crate::cpu::render::CpuDispatchTileFn =
				[<$name _cpu_dispatch_tile>];

			pub fn [<$name _cpu>](
				in_data: &after_effects::InData,
				in_layer: &after_effects::Layer,
				out_layer: &mut after_effects::Layer,
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), after_effects::Error> {
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
	};
}
