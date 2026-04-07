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

// Implementations for each variant
impl FromParam for i32 {
	fn extract(p: pr::Param) -> Option<Self> {
		if let pr::Param::Int32(v) = p { Some(v) } else { None }
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
	filter
		.param(param.to_index(), render_params.clip_time())
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

/// Declares a `#[repr(C)]` kernel params struct with auto-generated `from_gpu` and `from_cpu`.
///
/// # Extractors
///
/// | Extractor         | GPU                              | CPU                          |
/// |-------------------|----------------------------------|------------------------------|
/// | `float(V)`        | `get_param::<f32>`               | `params.float(V)?`           |
/// | `angle(V)`        | `get_param::<f32>`               | `params.angle(V)?`           |
/// | `color_r(V)`      | `get_param::<Pixel>.red as f32`  | `params.color(V)?.red as f32`|
/// | `color_g(V)`      | `.green`                         | `.green`                     |
/// | `color_b(V)`      | `.blue`                          | `.blue`                      |
/// | `color_a(V)`      | `.alpha`                         | `.alpha`                     |
/// | `point_pct_x(V)`  | `point.0 / 100.0`                | `point.0 / 100.0`           |
/// | `point_pct_y(V)`  | `point.1 / 100.0`                | `point.1 / 100.0`           |
/// | `checkbox(V)`     | `get_param::<bool> as u32`       | `params.checkbox(V)? as u32` |
///
/// Append `/ expr` or `* expr` after the extractor for a post-transform.
/// Fields without `= ...` are zero-initialized padding.
///
/// # Example
///
/// ```ignore
/// prgpu::kernel_params! {
///     VignetteParams for crate::params::Params {
///         tint_r:     f32 = [color_r(Tint) / 255.0];
///         scale_x:    f32 = [float(ScaleX)];
///         anchor_x:   f32 = [point_pct_x(Anchor)];
///         noise_phase:f32 = [angle(NoiseTimeOffset) * prgpu::params::DEG_TO_RAD];
///         _pad:       [f32; 3];
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
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            $( pub $field : $ty, )*
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
            ) -> ::core::result::Result<Self, ::after_effects::Error> {
                use $crate::params::CpuParams as _;
                Ok(Self {
                    $( $field: $crate::kernel_params!(@cpu $P, __params $(, $($spec)+)?), )*
                })
            }
        }
    };

    // GPU: transform wrappers

    // With / transform
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $ext:ident($v:ident) / $($t:tt)+) => {
        $crate::kernel_params!(@gpu_base $ext, $P, $f, $rp, $w, $h, $v) / $($t)+
    };
    // With * transform
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $ext:ident($v:ident) * $($t:tt)+) => {
        $crate::kernel_params!(@gpu_base $ext, $P, $f, $rp, $w, $h, $v) * $($t)+
    };
    // No transform
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $ext:ident($v:ident)) => {
        $crate::kernel_params!(@gpu_base $ext, $P, $f, $rp, $w, $h, $v)
    };
    // Padding (no spec)
    (@gpu $P:path, $f:ident, $rp:ident, $w:ident, $h:ident) => {
        Default::default()
    };

    // GPU: base extractors

    (@gpu_base float, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<f32, _>($f, <$P>::$v, $rp)
    };
    (@gpu_base angle, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<f32, _>($f, <$P>::$v, $rp)
    };
    (@gpu_base checkbox, $P:path, $f:ident, $rp:ident, $w:ident, $h:ident, $v:ident) => {
        $crate::params::get_param::<bool, _>($f, <$P>::$v, $rp) as u32
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

    // CPU: transform wrappers

    // With / transform
    (@cpu $P:path, $p:ident, $ext:ident($v:ident) / $($t:tt)+) => {
        $crate::kernel_params!(@cpu_base $ext, $P, $p, $v) / $($t)+
    };
    // With * transform
    (@cpu $P:path, $p:ident, $ext:ident($v:ident) * $($t:tt)+) => {
        $crate::kernel_params!(@cpu_base $ext, $P, $p, $v) * $($t)+
    };
    // No transform
    (@cpu $P:path, $p:ident, $ext:ident($v:ident)) => {
        $crate::kernel_params!(@cpu_base $ext, $P, $p, $v)
    };
    // Padding (no spec)
    (@cpu $P:path, $p:ident) => {
        Default::default()
    };

    // CPU: base extractors

    (@cpu_base float, $P:path, $p:ident, $v:ident) => {
        $p.float(<$P>::$v)?
    };
    (@cpu_base angle, $P:path, $p:ident, $v:ident) => {
        $p.angle(<$P>::$v)?
    };
    (@cpu_base checkbox, $P:path, $p:ident, $v:ident) => {
        $p.checkbox(<$P>::$v)? as u32
    };
    (@cpu_base color_r, $P:path, $p:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.red as f32
    }};
    (@cpu_base color_g, $P:path, $p:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.green as f32
    }};
    (@cpu_base color_b, $P:path, $p:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.blue as f32
    }};
    (@cpu_base color_a, $P:path, $p:ident, $v:ident) => {{
        let __c = $p.color(<$P>::$v)?;
        __c.alpha as f32
    }};
    (@cpu_base point_pct_x, $P:path, $p:ident, $v:ident) => {{
        let __pt = $p.point(<$P>::$v)?;
        __pt.0 / 100.0
    }};
    (@cpu_base point_pct_y, $P:path, $p:ident, $v:ident) => {{
        let __pt = $p.point(<$P>::$v)?;
        __pt.1 / 100.0
    }};
}
