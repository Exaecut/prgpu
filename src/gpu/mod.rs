use premiere::{self as pr};
use std::slice;

pub mod backends;
pub mod metrics;
pub mod render_properties;
pub mod scheduling;
pub mod shaders;

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

#[inline]
fn frames_as_slice<'a>(frames: *const pr::sys::PPixHand, frame_count: usize) -> Result<&'a [pr::sys::PPixHand], pr::Error> {
	if frames.is_null() || frame_count == 0 {
		return Err(pr::Error::Fail);
	}

	Ok(unsafe { slice::from_raw_parts(frames, frame_count) })
}

fn gpu_bytes_per_pixels(pixel_format: pr::PixelFormat) -> i32 {
	match pixel_format {
		pr::PixelFormat::GpuBgra4444_32f => 16, // float4
		pr::PixelFormat::GpuBgra4444_16f => 8,  // half4
		_ => panic!("Unsupported pixel format"),
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

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda", gpu_backend = "opencl", gpu_backend = "other")))]
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

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda", gpu_backend = "opencl", gpu_backend = "other")))]
	mod imp {
		compile_error!("Unsupported gpu_backend");
	}
}

pub mod fence {
	pub use imp::*;

	#[cfg(gpu_backend = "metal")]
	mod imp {
		pub use crate::gpu::backends::metal::fence::*;
	}

	#[cfg(gpu_backend = "cuda")]
	mod imp {
		pub use crate::gpu::backends::cuda::fence::*;
	}

	#[cfg(gpu_backend = "opencl")]
	mod imp {
		unimplemented!("OpenCL backend not yet implemented");
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda", gpu_backend = "opencl", gpu_backend = "other")))]
	mod imp {
		compile_error!("Unsupported gpu_backend");
	}
}
