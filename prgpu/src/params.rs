use std::{fmt::Debug, hash::Hash, ptr};

use after_effects::{InData, OutData, Parameters, log, sys::PF_Pixel};
use premiere::{self as pr};

use crate::types::Pixel;

pub trait SetupParams: Sized + Eq + Hash + Copy + Debug + Into<usize> {
    fn to_index(self) -> usize {
        self.into()
    }

    fn setup(
        params: &mut Parameters<Self>,
        in_data: InData,
        out_data: OutData,
    ) -> Result<(), after_effects::Error>;
}

pub trait FromParam: Sized {
    fn extract(p: pr::Param) -> Option<Self>;
}

// Implementations for each variant
impl FromParam for i32 {
    fn extract(p: pr::Param) -> Option<Self> {
        if let pr::Param::Int32(v) = p {
            Some(v)
        } else {
            None
        }
    }
}

impl FromParam for i64 {
    fn extract(p: pr::Param) -> Option<Self> {
        if let pr::Param::Int64(v) = p {
            Some(v)
        } else {
            None
        }
    }
}

impl FromParam for f32 {
    fn extract(p: pr::Param) -> Option<Self> {
        if let pr::Param::Float32(v) = p {
            Some(v)
        } else if let pr::Param::Float64(v) = p {
            log::warn!("Parameter {p:?} was f64, converting to f32 (may lose precision)");
            Some(v as f32)
        } else {
            None
        }
    }
}

impl FromParam for f64 {
    fn extract(p: pr::Param) -> Option<Self> {
        if let pr::Param::Float64(v) = p {
            Some(v)
        } else {
            None
        }
    }
}

impl FromParam for bool {
    fn extract(p: pr::Param) -> Option<Self> {
        if let pr::Param::Bool(v) = p {
            Some(v)
        } else {
            None
        }
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
        if let pr::Param::Point(v) = p {
            Some((v.x as f32, v.y as f32))
        } else {
            None
        }
    }
}

impl FromParam for (f64, f64) {
    fn extract(p: premiere::Param) -> Option<Self> {
        if let pr::Param::Point(v) = p {
            Some((v.x, v.y))
        } else {
            None
        }
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

pub fn get_param<T: FromParam + Default, Params: SetupParams>(
    filter: &pr::GpuFilterData,
    param: Params,
    render_params: &pr::RenderParams,
) -> T {
    filter
        .param(param.to_index(), render_params.clip_time())
        .ok()
        .and_then(T::extract)
        .unwrap_or_default()
}
