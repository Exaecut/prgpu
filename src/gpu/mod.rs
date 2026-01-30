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
