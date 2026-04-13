#[cfg(gpu_backend = "metal")]
pub mod metal;

#[cfg(gpu_backend = "cuda")]
pub mod cuda;

use crate::types::Configuration;

/// Routes a kernel dispatch to the active GPU backend.
///
/// `shader_src` carries the backend-native primary artifact:
/// - Metal:  pre-expanded `.metal` source (managed by `metal::run` / `metal::pipeline`)
/// - CUDA:   build-time PTX string
/// - OpenCL: build-time `.cl` source
///
/// `shader_src_f16` carries the CUDA half-precision PTX variant.
/// It is an empty string on non-CUDA builds and is ignored by Metal/OpenCL.
pub fn dispatch_kernel<UP>(
    config: &Configuration,
    user_params: UP,
    shader_src: &'static str,
    shader_src_f16: &'static str,
    entry: &'static str,
) -> Result<(), &'static str>
{
    #[cfg(gpu_backend = "metal")]
    {
        let _ = shader_src_f16; // Metal compiles f32 and f16 from the same source internally
        return metal::run::<UP>(config, user_params, shader_src, entry);
    }

    #[cfg(gpu_backend = "cuda")]
    {
        return cuda::run::<UP>(config, user_params, shader_src, shader_src_f16, entry);
    }

    #[cfg(gpu_backend = "opencl")]
    {
        let _ = shader_src_f16;
        unimplemented!("OpenCL backend not yet implemented");
        // return opencl::run::<UP>(config, user_params, shader_src, entry);
    }

    #[allow(unreachable_code)]
    Err("no GPU backend enabled")
}
