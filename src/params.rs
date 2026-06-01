use std::{any::TypeId, collections::HashMap, fmt::Debug, hash::Hash, ptr, sync::{OnceLock, RwLock}};

use after_effects::{InData, OutData, Parameters, sys::PF_Pixel};
use premiere::{self as pr};

use crate::types::Pixel;

pub const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

pub trait SetupParams: Sized + Eq + Hash + Copy + Debug + Into<usize> + 'static {
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

/// Per-`Params` map of enum discriminant → host param index, captured from the
/// AE `Parameters` registration order during param setup.
///
/// The Premiere GPU path reads params positionally via `GpuFilterData::param(i)`,
/// where `i` is the registration-order index — NOT the enum discriminant. They
/// only coincide when params are added in discriminant order; an effect that
/// registers a param out of order (every effect registers its license button
/// first, which is the highest discriminant) shifts every other param by one.
/// Capturing the real map keeps the GPU path aligned with the CPU/AE path, which
/// already resolves through `Parameters::index`.
fn gpu_param_index_registry() -> &'static RwLock<HashMap<TypeId, HashMap<usize, usize>>> {
    static REG: OnceLock<RwLock<HashMap<TypeId, HashMap<usize, usize>>>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Record the discriminant → host-index map for `P`. Called once per param
/// setup by the adapter, after `Effect::params` has registered everything.
pub fn register_gpu_param_indices<P: SetupParams>(map: HashMap<usize, usize>) {
    if let Ok(mut reg) = gpu_param_index_registry().write() {
        reg.insert(TypeId::of::<P>(), map);
    }
}

/// Resolve a discriminant to its host param index for `P`. Falls back to the
/// discriminant itself when no map is registered, preserving legacy behavior
/// and keeping `from_gpu` unit tests (which never run param setup) unchanged.
fn gpu_param_index<P: SetupParams>(discriminant: usize) -> usize {
    gpu_param_index_registry()
        .read()
        .ok()
        .and_then(|reg| reg.get(&TypeId::of::<P>()).and_then(|m| m.get(&discriminant).copied()))
        .unwrap_or(discriminant)
}

pub fn get_param<T: FromParam + Default, Params: SetupParams>(filter: &pr::GpuFilterData, param: Params, render_params: &pr::RenderParams) -> T {
    let discriminant = param.to_index();
    let idx = gpu_param_index::<Params>(discriminant);
    let clip = render_params.clip_time();
    match filter.param(idx, clip) {
        Ok(p) => match T::extract(p) {
            Some(v) => v,
            None => {
                #[cfg(debug_assertions)]
                after_effects::log::warn!(
                    "[params] discriminant {discriminant} (host idx {idx}): present but not the variant this kernel field expects; substituting Default (0). A zeroed param (e.g. angle=0) makes the effect a no-op / passthrough."
                );
                T::default()
            }
        },
        Err(_e) => {
            #[cfg(debug_assertions)]
            after_effects::log::warn!("[params] discriminant {discriminant} (host idx {idx}): lookup failed ({_e:?}); substituting Default (0).");
            T::default()
        }
    }
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
