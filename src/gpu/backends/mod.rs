#[cfg(gpu_backend = "metal")]
pub mod metal;

#[cfg(gpu_backend = "cuda")]
pub mod cuda;

use crate::types::Configuration;

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
        let _ = shader_src_f16;
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
