//! `kernel_params!` and `declare_kernel!` macros co-located.
//!
//! Both are `#[macro_export]`-rooted so users invoke them as
//! `prgpu::kernel_params!` and `prgpu::declare_kernel!` regardless of where
//! the source lives. They live here, next to `Kernel<P>` and
//! `KernelParams`, because that's where the concept they generate code for
//! lives.

/// Declare a `gpu_struct`-laid-out kernel params struct with auto-generated
/// `from_gpu` / `from_cpu` / `from_context` and a [`crate::kernel::params::KernelParams`] impl.
///
/// Layout, alignment, padding, and ABI checks are delegated to
/// `#[prgpu::gpu_struct]`; the macro adds the host-side parameter extractors
/// and the `KernelParams` marker the dispatcher relies on.
///
/// Extractors: `float`, `angle`, `color_r/g/b/a`, `point_pct_x/y`, `checkbox`, `popup`.
/// Append `/ expr` or `* expr` after an extractor for a post-transform.
///
/// # Popup contract
///
/// `popup(V)` always hands the kernel a 0-based selected-index (`0` = first option)
/// regardless of host. Author popup-driven enums as plain 0-based slang enums:
///
/// ```cpp
/// enum SampleDistribution : uint { Linear = 0, Exponential = 1, Gaussian = 2 };
/// ```
///
/// AE PF CPU values (1-based) get a `saturating_sub(1)`; Premiere GPU values
/// (already 0-based) pass through clamped to ≥ 0.
///
/// Fields without `= ...` are zero-initialized padding; frame-level state belongs
/// in `FrameParams`, not per-pass kernel params.
///
/// ```ignore
/// prgpu::kernel_params! {
///     VignetteParams for crate::params::Params {
///         tint_r:     f32 = [color_r(Tint) / 255.0];
///         scale_x:    f32 = [float(ScaleX)];
///         anchor_x:   f32 = [point_pct_x(Anchor)];
///         noise_phase:f32 = [angle(NoiseTimeOffset) * prgpu::params::DEG_TO_RAD];
///         _pad0:      f32;
///     }
/// }
/// ```
#[macro_export]
macro_rules! kernel_params {
    (
        $name:ident for $P:path {
            $( $field:ident : $ty:ty $(= [$($spec:tt)+])? ; )*
        }
    ) => {
        #[$crate::gpu_struct]
        pub struct $name {
            $( pub $field : $ty, )*
        }

        impl $crate::KernelParams for $name {
            const SIZE: usize = <$name>::SIZE;
            const ALIGN: usize = <$name>::ALIGN;
        }

        impl $name {
            pub fn from_gpu(
                __filter: &::premiere::GpuFilterData,
                __rp: &::premiere::RenderParams,
                __width: f32,
                __height: f32,
            ) -> Self {
                Self {
                    $( $field: $crate::kernel_params!(@gpu $P, __filter, __rp, __width, __height $(, $($spec)+)?), )*
                }
            }

            pub fn from_cpu(
                __params: &::after_effects::Parameters<'_, $P>,
                __width: f32,
                __height: f32,
                __is_premiere: bool,
            ) -> ::core::result::Result<Self, ::after_effects::Error> {
                use $crate::params::CpuParams as _;
                Ok(Self {
                    $( $field: $crate::kernel_params!(@cpu $P, __params, __width, __height, __is_premiere $(, $($spec)+)?), )*
                })
            }

            /// Host-agnostic extraction. Picks `from_cpu` (AE) or `from_gpu`
            /// (Premiere GPU) based on the active `FrameDataContext`.
            pub fn from_context(
                __ctx: &$crate::effect::FrameDataContext<'_, $P>,
            ) -> ::core::result::Result<Self, ::after_effects::Error> {
                __ctx.extract_kernel_params(Self::from_cpu, Self::from_gpu)
            }
        }
    };


    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $ext:ident($v:ident) / $($t:tt)+) => {
        $crate::kernel_params!(@gpu_base $ext, $P, $f, $rp, $w, $h, $v) / $($t)+
    };
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $ext:ident($v:ident) * $($t:tt)+) => {
        $crate::kernel_params!(@gpu_base $ext, $P, $f, $rp, $w, $h, $v) * $($t)+
    };
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $ext:ident($v:ident)) => {
        $crate::kernel_params!(@gpu_base $ext, $P, $f, $rp, $w, $h, $v)
    };
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident) => {
        Default::default()
    };


    (@gpu_base float, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<f32, _>($f, <$P>::$v, $rp)
    };
    (@gpu_base angle, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<f32, _>($f, <$P>::$v, $rp)
    };
    (@gpu_base checkbox, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<bool, _>($f, <$P>::$v, $rp) as u32
    };
    // Premiere GPU's ParamBuffer returns popups already 0-based; clamp negatives to 0 defensively.
    (@gpu_base popup, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<i32, _>($f, <$P>::$v, $rp).max(0) as u32
    };
    (@gpu_base color_r, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __c: $crate::types::Pixel = $crate::params::get_param($f, <$P>::$v, $rp);
        __c.red as f32
    }};
    (@gpu_base color_g, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __c: $crate::types::Pixel = $crate::params::get_param($f, <$P>::$v, $rp);
        __c.green as f32
    }};
    (@gpu_base color_b, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __c: $crate::types::Pixel = $crate::params::get_param($f, <$P>::$v, $rp);
        __c.blue as f32
    }};
    (@gpu_base color_a, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __c: $crate::types::Pixel = $crate::params::get_param($f, <$P>::$v, $rp);
        __c.alpha as f32
    }};
    (@gpu_base point_pct_x, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __p: (f32, f32) = $crate::params::get_param($f, <$P>::$v, $rp);
        __p.0
    }};
    (@gpu_base point_pct_y, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __p: (f32, f32) = $crate::params::get_param($f, <$P>::$v, $rp);
        __p.1
    }};
    (@gpu_base point_px_x, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __p: (f32, f32) = $crate::params::get_param($f, <$P>::$v, $rp);
        __p.0 * $w
    }};
    (@gpu_base point_px_y, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {{
        let __p: (f32, f32) = $crate::params::get_param($f, <$P>::$v, $rp);
        __p.1 * $h
    }};


    (@cpu $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $ext:ident($v:ident) / $($t:tt)+) => {
        $crate::kernel_params!(@cpu_base $ext, $P, $p, $w, $h, $is_pr, $v) / $($t)+
    };
    (@cpu $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $ext:ident($v:ident) * $($t:tt)+) => {
        $crate::kernel_params!(@cpu_base $ext, $P, $p, $w, $h, $is_pr, $v) * $($t)+
    };
    (@cpu $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $ext:ident($v:ident)) => {
        $crate::kernel_params!(@cpu_base $ext, $P, $p, $w, $h, $is_pr, $v)
    };
    (@cpu $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident) => {
        Default::default()
    };

    // CPU base extractors: $w/$h normalize points, $is_pr is reserved for future Premiere CPU R↔B swaps.

    (@cpu_base float, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {
        $p.float(<$P>::$v)?
    };
    (@cpu_base angle, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {
        $p.angle(<$P>::$v)?
    };
    (@cpu_base checkbox, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {
        $p.checkbox(<$P>::$v)? as u32
    };
    // AE PF popups are 1-based at the SDK; subtract 1 so the kernel sees the
    // same 0-based index as Premiere GPU. `saturating_sub` guards `value() == 0`.
    (@cpu_base popup, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {
        ($p.popup(<$P>::$v)? as u32).saturating_sub(1)
    };
    // Premiere PF_Pixel is BGRA: `.red` actually holds B and `.blue` holds R.
    // AE uses ARGB (correct).
    (@cpu_base color_r, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.red as f32
    }};
    (@cpu_base color_g, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.green as f32
    }};
    (@cpu_base color_b, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.blue as f32
    }};
    (@cpu_base color_a, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.alpha as f32
    }};
    // AE/Premiere `as_point().value()` returns pixel coordinates; normalize to UV
    // by dividing by layer dims. (GPU path receives pre-normalized values.)
    (@cpu_base point_pct_x, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __pt = $p.point(<$P>::$v)?;
        __pt.0 / $w
    }};
    (@cpu_base point_pct_y, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __pt = $p.point(<$P>::$v)?;
        __pt.1 / $h
    }};
    (@cpu_base point_px_x, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __pt = $p.point(<$P>::$v)?;
        __pt.0
    }};
    (@cpu_base point_px_y, $P:path, $p:ident, $w:ident, $h:ident, $is_pr:ident, $v:ident) => {{
        let __pt = $p.point(<$P>::$v)?;
        __pt.1
    }};
}

/// Declare a GPU kernel (CUDA + Metal) and its CPU fallback dispatch.
///
/// `declare_kernel!(vignette, VignetteParams);` emits a per-kernel module
/// (`vignette::*`) plus deprecated top-level aliases for backward compat:
///
/// ```ignore
/// // New API:
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
/// The generated `kernel()` constructor returns a [`crate::kernel::Kernel`] that
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
