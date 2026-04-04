use std::slice;

use after_effects::log;
use premiere::{self as pr, PixelFormat, Property};

use crate::PrRect;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GPUFramework {
    Metal,
    Cuda,
    OpenCL,
    Other(u32),
}

impl GPUFramework {
    pub fn from_premiere(v: u32) -> Self {
        match v {
            0 => Self::Cuda,
            1 => Self::OpenCL,
            2 => Self::Metal,
            _ => Self::Other(v),
        }
    }
}

pub mod backends;
pub mod shaders;

#[derive(Debug, Clone)]
pub struct GPURenderProperties {
    pub progress: f32,
    pub gpu_index: u32,
    pub pixel_format: PixelFormat,
    pub half_precision: bool,
    pub bounds: after_effects::Rect,
    pub output_frame: pr::sys::PPixHand,
    pub frames: (pr::sys::PPixHand, pr::sys::PPixHand),
    pub bytes_per_pixel: i32,
}

#[inline]
fn frames_as_slice<'a>(
    frames: *const pr::sys::PPixHand,
    frame_count: usize,
) -> &'a [pr::sys::PPixHand] {
    assert!(!frames.is_null(), "frames pointer was null");
    unsafe { slice::from_raw_parts(frames, frame_count) }
}

fn gpu_bytes_per_pixels(pixel_format: pr::PixelFormat) -> i32 {
    match pixel_format {
        pr::PixelFormat::GpuBgra4444_32f => 16, // float4
        pr::PixelFormat::GpuBgra4444_16f => 8,  // half4
        _ => panic!("Unsupported pixel format"),
    }
}

impl GPURenderProperties {
    pub fn new(
        filter: &premiere::GpuFilterData,
        render_params: premiere::RenderParams,
        frames: *const premiere::sys::PPixHand,
        frame_count: usize,
    ) -> Result<Self, premiere::Error> {
        unsafe {
            (*filter.instance_ptr).outIsRealtime = 1;
        }

        if frame_count < 2 || frames.is_null() {
            log::error!("Invalid frame count or null frames pointer");
            return Err(pr::Error::Fail);
        }

        let frames = frames_as_slice(frames, frame_count);
        let outgoing = frames[0];
        let incoming = frames[1];

        if incoming.is_null() || outgoing.is_null() {
            log::error!("Incoming or outgoing frame is null");
            return Err(pr::Error::Fail);
        }

        let transition_suite = pr::suites::Transition::new();
        let key = if transition_suite.is_ok() {
            Property::Transition_Duration
        } else {
            Property::Effect_EffectDuration
        };

        let progress = match filter.property(key) {
            Ok(pr::PropertyData::Int64(d)) if d != 0 => render_params.clip_time() as f64 / d as f64,
            Ok(pr::PropertyData::Time(d)) if d != 0 => render_params.clip_time() as f64 / d as f64,
            Ok(property_data) => {
                log::error!("Retrieved unexpected property data: {property_data:?}");
                return Err(pr::Error::InvalidParms);
            }
            Err(error) => {
                log::error!("Failed to get transition duration: {error:?}");
                return Err(pr::Error::InvalidParms);
            }
        } as f32;

        let properties = if !incoming.is_null() {
            incoming
        } else {
            outgoing
        };

        let gpu_index = match filter.gpu_device_suite.gpu_ppix_device_index(properties) {
            Ok(index) => index,
            Err(_) => {
                log::error!("Failed to get GPU device index for properties");
                return Err(pr::Error::InvalidParms);
            }
        };

        let pixel_format = match filter.ppix_suite.pixel_format(properties) {
            Ok(format) => format,
            Err(_) => {
                log::error!("Failed to get pixel format for properties");
                return Err(pr::Error::InvalidParms);
            }
        };

        let half_precision = pixel_format != pr::PixelFormat::GpuBgra4444_32f;
        let bounds: after_effects::Rect =
            after_effects::Rect::from(PrRect::from(filter.ppix_suite.bounds(properties).unwrap()));

        let width = bounds.width();
        let height = bounds.height();

        let (par_numerator, par_denominator) =
            match filter.ppix_suite.pixel_aspect_ratio(properties) {
                Ok((num, den)) => (num as i32, den as i32),
                Err(_) => {
                    log::error!("Failed to get pixel aspect ratio for properties");
                    return Err(pr::Error::InvalidParms);
                }
            };

        let field_type = match filter.ppix2_suite.field_order(properties) {
            Ok(field_type) => field_type,
            Err(_) => {
                log::error!("Failed to get field type for properties");
                return Err(pr::Error::InvalidParms);
            }
        };

        let output_frame = match filter.gpu_device_suite.create_gpu_ppix(
            gpu_index,
            pixel_format,
            width,
            height,
            par_numerator,
            par_denominator,
            field_type,
        ) {
            Ok(frame) => frame,
            Err(_) => {
                log::error!("Failed to create GPU PPix");
                return Err(pr::Error::InvalidParms);
            }
        };

        if output_frame.is_null() {
            log::error!("Output frame is null");
            return Err(pr::Error::Fail);
        }

        let bytes_per_pixel = gpu_bytes_per_pixels(pixel_format);

        Ok(GPURenderProperties {
            progress,
            gpu_index,
            pixel_format,
            half_precision,
            bounds,
            output_frame,
            bytes_per_pixel,
            frames: (incoming, outgoing),
        })
    }
}

pub mod buffer {
    pub use imp::*;

    #[cfg(gpu_backend = "metal")]
    mod imp {
        pub use crate::gpu::backends::metal::buffer::*;
    }

    #[cfg(gpu_backend = "cuda")]
    mod imp {
        pub use crate::gpu::backends::cuda::buffer::*;
    }

    #[cfg(gpu_backend = "opencl")]
    mod imp {
        unimplemented!("OpenCL backend not yet implemented");
    }

    #[cfg(not(any(
        gpu_backend = "metal",
        gpu_backend = "cuda",
        gpu_backend = "opencl",
        gpu_backend = "other"
    )))]
    mod imp {
        compile_error!("Unsupported gpu_backend");
    }
}

pub mod pipeline {
    pub use imp::*;

    #[cfg(gpu_backend = "metal")]
    mod imp {
        pub use crate::gpu::backends::metal::pipeline::*;
    }

    #[cfg(gpu_backend = "cuda")]
    mod imp {
        pub use crate::gpu::backends::cuda::pipeline::*;
    }

    #[cfg(gpu_backend = "opencl")]
    mod imp {
        unimplemented!("OpenCL backend not yet implemented");
    }

    #[cfg(not(any(
        gpu_backend = "metal",
        gpu_backend = "cuda",
        gpu_backend = "opencl",
        gpu_backend = "other"
    )))]
    mod imp {
        compile_error!("Unsupported gpu_backend");
    }
}
