pub mod helpers;

/// Declares a GPU kernel (CUDA & Metal) + CPU fallback dispatch.
///
/// Usage: `declare_kernel!(vignette, VignetteParams);`
///
/// Generates:
/// - `const VIGNETTE_SHADER_SRC` — the primary shader artifact for the active backend:
///   - Metal: embedded `.metallib` bytes (`include_shader!(name, metal)`)
///   - CUDA: embedded `.ptx` source (`include_shader!(name, cuda)`)
/// - `const VIGNETTE_KERNEL_ENTRY_POINT`
/// - `pub unsafe fn vignette(config, user_params)` (GPU dispatch)
/// - `pub fn vignette_cpu(in_data, in_layer, out_layer, config, user_params)` (CPU dispatch)
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

				/// Tile entry — loops `y ∈ [y0, y1) × x ∈ [0, width)` in C.
				pub fn [<$name _cpu_dispatch_tile>](
					y0: u32,
					y1: u32,
					width: u32,
					buffers: *const *const std::ffi::c_void,
					transition_params: *const std::ffi::c_void,
					user_params: *const std::ffi::c_void,
				);
			}

			/// Typed pointer to the per-pixel CPU dispatch function.
			#[allow(non_upper_case_globals)]
			pub const [<$name:upper _CPU_DISPATCH>]: $crate::cpu::render::CpuDispatchFn =
				[<$name _cpu_dispatch>];

			/// Typed pointer to the tile CPU dispatch function.
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
