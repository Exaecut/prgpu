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
