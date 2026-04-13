/// Declares a GPU kernel (CUDA & Metal) + CPU fallback dispatch.
///
/// Usage: `declare_kernel!(vignette, VignetteParams);`
///
/// Generates:
/// - `const VIGNETTE_SHADER_SRC` — the primary shader artifact for the active backend:
///   - Metal: embedded `.metal` source (`include_shader!(name, metal)`)
///   - CUDA: embedded `.ptx` source (`include_shader!(name, cuda)`)
///   - OpenCL: embedded `.cl` source (`include_shader!(name, opencl)`)
/// - `const VIGNETTE_SHADER_SRC_F16` — CUDA half-precision PTX; empty string on non-CUDA
/// - `const VIGNETTE_KERNEL_ENTRY_POINT`
/// - `pub unsafe fn vignette(config, user_params)` (GPU dispatch)
/// - `pub fn vignette_cpu(in_data, in_layer, out_layer, config, user_params)` (CPU dispatch)
///
/// Under `shader_hotreload`:
/// - The GPU function registers the effect's shader directory once via a `Once` guard,
///   then routes every dispatch through the hot-reload pipeline (NVRTC / Metal runtime
///   compiler) with automatic fallback to the embedded build-time artifact.
/// - The CPU function registers its shader directory the same way and resolves the
///   dispatch function pointer through `cpu::pipeline::get_dispatch_fn()`, which
///   compiles the `.vekl` to a shared library on first use (or after `hot_reload()`)
///   and falls back to the statically-linked symbol on any error.
///
/// Without `shader_hotreload`:
/// - GPU dispatch uses the embedded build-time artifact directly (zero indirection).
/// - CPU dispatch calls the statically-linked `{name}_cpu_dispatch` symbol directly
///   via the no-op `cpu::pipeline::get_dispatch_fn()` stub that returns the fallback.
#[macro_export]
macro_rules! declare_kernel {
	($name:ident, $user_params_ty:ty) => {
		$crate::paste::paste! {
			/// Primary shader source for the active GPU backend.
			/// Selects `.metal`, `.ptx`, or `.cl` at compile time via gpu_backend cfg.
			#[allow(non_upper_case_globals)]
			const [<$name:upper _SHADER_SRC>]: &str = {
				#[cfg(gpu_backend = "metal")]
				{ $crate::include_shader!($name, metal) }

				#[cfg(gpu_backend = "cuda")]
				{ $crate::include_shader!($name, cuda) }

				#[cfg(gpu_backend = "opencl")]
				{ $crate::include_shader!($name, opencl) }

				#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda", gpu_backend = "opencl")))]
				{ "" }
			};
		}

		$crate::paste::paste! {
			/// Half-precision CUDA PTX variant. Empty string on non-CUDA backends.
			#[allow(non_upper_case_globals)]
			const [<$name:upper _SHADER_SRC_F16>]: &str = {
				#[cfg(gpu_backend = "cuda")]
				{ $crate::include_shader!($name, cuda, halfprecision) }

				#[cfg(not(gpu_backend = "cuda"))]
				{ "" }
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
			///
			/// Under `shader_hotreload`, the dispatch function pointer is resolved
			/// through `cpu::pipeline::get_dispatch_fn()` which compiles the `.vekl`
			/// source at runtime and returns the hot-reloaded symbol, falling back to
			/// the statically-linked `{name}_cpu_dispatch` on any error.
			pub fn [<$name _cpu>](
				in_data: &after_effects::InData,
				in_layer: &after_effects::Layer,
				out_layer: &mut after_effects::Layer,
				config: &$crate::types::Configuration,
				user_params: $user_params_ty,
			) -> Result<(), after_effects::Error> {
				// Register shader directories once on the first CPU render frame.
				// The `Once` guard is local to this function so each kernel registers
				// independently; all kernels in the same effect share the same dirs.
				#[cfg(shader_hotreload)]
				{
					static SHADER_DIR_INIT: std::sync::Once = std::sync::Once::new();
					SHADER_DIR_INIT.call_once(|| {
						let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
						let shader_dir = manifest.join("shaders");
						let vekl_dir = manifest.join("..").join("vekl");
						$crate::cpu::pipeline::set_shader_dirs(shader_dir, vec![vekl_dir]);
					});
				}

				// Resolve the dispatch function pointer.
				// Under shader_hotreload: get_dispatch_fn() looks up the runtime-compiled
				// shared library, compiling it on first use, and falls back to the static
				// symbol on any error.
				// Without shader_hotreload: the stub returns static_fallback unchanged —
				// zero overhead, compiles down to a direct call.
				let dispatch_fn = $crate::cpu::pipeline::get_dispatch_fn(
					stringify!($name),
					[<$name _cpu_dispatch>],
				);

				$crate::cpu::render::render_cpu(
					in_data,
					in_layer,
					out_layer,
					config,
					dispatch_fn,
					&user_params,
				)
			}
		}
	};
}
