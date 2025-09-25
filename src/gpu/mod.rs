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

pub mod buffer {
	pub use imp::*;

	#[cfg(all(feature = "metal", target_os = "macos"))]
	mod imp {
		pub use crate::gpu::backends::metal::buffer::*;
	}

	#[cfg(all(feature = "cuda", target_os = "windows"))]
	mod imp {
		pub use crate::gpu::backends::cuda::buffer::*;
	}
}

pub mod pipeline {
	pub use imp::*;

	#[cfg(all(feature = "metal", target_os = "macos"))]
	mod imp {
		pub use crate::gpu::backends::metal::pipeline::*;
	}

	#[cfg(all(feature = "cuda", target_os = "windows"))]
	mod imp {
		pub use crate::gpu::backends::cuda::pipeline::*;
	}
}
