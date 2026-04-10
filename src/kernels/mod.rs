/// Declares a GPU kernel (CUDA & Metal) + CPU fallback dispatch.
///
/// Usage: `declare_kernel!(vignette, VignetteParams);`
///
/// Generates:
/// - `const VIGNETTE_SHADER_SRC` (embedded PTX)
/// - `const VIGNETTE_SHADER_SRC_F16` (embedded PTX, half-precision)
/// - `const VIGNETTE_KERNEL_ENTRY_POINT`
/// - `pub unsafe fn vignette(config, user_params)` (GPU dispatch)
/// - `pub fn vignette_cpu(in_data, in_layer, out_layer, config, user_params)` (CPU via iterate_with/rayon)
///
/// Under `shader_hotreload`, the generated dispatch function auto-registers
/// the effect's shader directory on first call. No manual setup required.
#[macro_export]
macro_rules! declare_kernel {
	($name:ident, $user_params_ty:ty) => {
		$crate::paste::paste! {
			#[allow(non_upper_case_globals)]
			const [<$name:upper _SHADER_SRC>]: &str = $crate::include_shader!($name);
		}

		$crate::paste::paste! {
			#[allow(non_upper_case_globals)]
			const [<$name:upper _SHADER_SRC_F16>]: &str = $crate::include_shader!($name, halfprecision);
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
				#[cfg(shader_hotreload)]
				{
					static SHADER_DIR_INIT: std::sync::Once = std::sync::Once::new();
					SHADER_DIR_INIT.call_once(|| {
						let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
						let shader_dir = manifest.join("shaders");
						let vekl_dir = manifest.join("..").join("vekl");
						$crate::gpu::pipeline::set_shader_dirs(shader_dir, vec![vekl_dir]);
					});
				}

				$crate::backends::dispatch_kernel::<$user_params_ty>(
					config,
					user_params,
					[<$name:upper _SHADER_SRC>],
					[<$name:upper _SHADER_SRC_F16>],
					[<$name:upper _KERNEL_ENTRY_POINT>],
				)
			}
		}

		$crate::paste::paste! {
			unsafe extern "C" {
				fn [<$name _cpu_dispatch>](
					gid_x: u32,
					gid_y: u32,
					buffers: *const *const std::ffi::c_void,
					transition_params: *const std::ffi::c_void,
					user_params: *const std::ffi::c_void,
				);
			}

			/// CPU dispatch via `iterate_with` (AE) or rayon (Premiere).
			///
			/// BPP is auto-detected from the output Layer. Configuration provides
			/// buffer pointers and pitches; when its dimensions match the output Layer,
			/// AE's multi-threaded iterate suites are used, otherwise rayon is used
			/// (e.g. for blur intermediate buffers with downsampled dimensions).
			pub fn [<$name _cpu>](
				in_data: &after_effects::InData,
				in_layer: &after_effects::Layer,
				out_layer: &mut after_effects::Layer,
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), after_effects::Error> {
				$crate::cpu::render::render_cpu(
					in_data,
					in_layer,
					out_layer,
					config,
					[<$name _cpu_dispatch>],
					&user_params,
				)
			}
		}
	};
}
