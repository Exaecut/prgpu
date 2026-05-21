use std::{fmt::Debug, hash::Hash, ptr};

use after_effects::{InData, OutData, Parameters, sys::PF_Pixel};
use premiere::{self as pr};

use crate::types::Pixel;

pub const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

pub trait SetupParams: Sized + Eq + Hash + Copy + Debug + Into<usize> {
	fn to_index(self) -> usize {
		self.into()
	}

	fn setup(params: &mut Parameters<Self>, in_data: InData, out_data: OutData) -> Result<(), after_effects::Error>;
}

pub trait FromParam: Sized {
	fn extract(p: pr::Param) -> Option<Self>;
}

impl FromParam for i32 {
	fn extract(p: pr::Param) -> Option<Self> {
		match p {
			pr::Param::Int32(v) => Some(v),
			// Premiere GPU returns popups as Float32; coerce instead of failing.
			pr::Param::Float32(v) => Some(v as i32),
			_ => None,
		}
	}
}

impl FromParam for i64 {
	fn extract(p: pr::Param) -> Option<Self> {
		if let pr::Param::Int64(v) = p { Some(v) } else { None }
	}
}

impl FromParam for f32 {
	fn extract(p: pr::Param) -> Option<Self> {
		if let pr::Param::Float32(v) = p {
			Some(v)
		} else if let pr::Param::Float64(v) = p {
			Some(v as f32)
		} else {
			None
		}
	}
}

impl FromParam for f64 {
	fn extract(p: pr::Param) -> Option<Self> {
		if let pr::Param::Float64(v) = p { Some(v) } else { None }
	}
}

impl FromParam for bool {
	fn extract(p: pr::Param) -> Option<Self> {
		if let pr::Param::Bool(v) = p { Some(v) } else { None }
	}
}

impl FromParam for u32 {
	fn extract(p: pr::Param) -> Option<Self> {
		if let pr::Param::Int32(v) = p {
			if v >= 0 { Some(v as u32) } else { None }
		} else {
			None
		}
	}
}

impl FromParam for (f32, f32) {
	fn extract(p: premiere::Param) -> Option<Self> {
		if let pr::Param::Point(v) = p { Some((v.x as f32, v.y as f32)) } else { None }
	}
}

impl FromParam for (f64, f64) {
	fn extract(p: premiere::Param) -> Option<Self> {
		if let pr::Param::Point(v) = p { Some((v.x, v.y)) } else { None }
	}
}

impl FromParam for Pixel {
	fn extract(p: premiere::Param) -> Option<Self> {
		match p {
			pr::Param::MemoryPtr(v) => {
				if v.is_null() {
					None
				} else {
					let pf_pixel = unsafe { ptr::read(v as *const PF_Pixel) };
					Some(Pixel::from_pf_pixel(pf_pixel))
				}
			}
			pr::Param::Int32(v) => Some(Pixel::from_bytes32(v as u32)),
			pr::Param::Int64(v) => {
				#[cfg(debug_assertions)]
				{
					Pixel::debug_print_color(v);
				}

				Some(Pixel::from_u64_color(v as u64))
			}
			_ => None,
		}
	}
}

pub fn get_param<T: FromParam + Default, Params: SetupParams>(filter: &pr::GpuFilterData, param: Params, render_params: &pr::RenderParams) -> T {
    let idx = param.to_index();
    let clip = render_params.clip_time();
    filter
        .param(idx, clip)
        .ok()
        .and_then(T::extract)
        .unwrap_or_default()
}

pub trait CpuParams<P: SetupParams> {
	fn float(&self, p: P) -> Result<f32, after_effects::Error>;
	fn angle(&self, p: P) -> Result<f32, after_effects::Error>;
	fn color(&self, p: P) -> Result<PF_Pixel, after_effects::Error>;
	fn point(&self, p: P) -> Result<(f32, f32), after_effects::Error>;
	fn checkbox(&self, p: P) -> Result<bool, after_effects::Error>;
	fn popup(&self, p: P) -> Result<i32, after_effects::Error>;
}

impl<P: SetupParams> CpuParams<P> for Parameters<'_, P> {
	fn float(&self, p: P) -> Result<f32, after_effects::Error> {
		Ok(self.get(p)?.as_float_slider()?.value() as f32)
	}

	fn angle(&self, p: P) -> Result<f32, after_effects::Error> {
		Ok(self.get(p)?.as_angle()?.value())
	}

	fn color(&self, p: P) -> Result<PF_Pixel, after_effects::Error> {
		Ok(self.get(p)?.as_color()?.value())
	}

	fn point(&self, p: P) -> Result<(f32, f32), after_effects::Error> {
		let v = self.get(p)?.as_point()?.value();
		Ok((v.0, v.1))
	}

	fn checkbox(&self, p: P) -> Result<bool, after_effects::Error> {
		Ok(self.get(p)?.as_checkbox()?.value())
	}

	fn popup(&self, p: P) -> Result<i32, after_effects::Error> {
		Ok(self.get(p)?.as_popup()?.value())
	}
}

/// Declare a `gpu_struct`-laid-out kernel params struct with auto-generated
/// `from_gpu` / `from_cpu` and a [`crate::KernelParams`] impl.
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
/// (already 0-based) pass through clamped to ≥ 0. Use `prgpu::ui::add_blend_mode_param`
/// to ship a popup whose 0-based values match the vekl `BLEND_*` constants.
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
    // Premiere GPU's ParamBuffer returns popups already 0-based (Multiply at popup pos 2 arrives as 1); clamp negatives to 0 defensively.
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

    // CPU base extractors: $w/$h normalize points, $is_pr drives Premiere R↔B color swap.

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
}
