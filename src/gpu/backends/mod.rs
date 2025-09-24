#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

use crate::types::Configuration;

pub fn dispatch_kernel<UP>(
    config: &Configuration,
    user_params: UP,
    shader_src: &'static str,
    entry: &'static str,
) -> Result<(), &'static str>
{
    #[cfg(feature = "metal")]
    {
        return metal::run::<UP>(config, user_params, shader_src, entry);
    }
    #[cfg(all(not(feature = "metal"), feature = "cuda"))]
    {
        return cuda::run::<UP>(config, user_params, shader_src, entry);
    }

    #[allow(unreachable_code)]
    Err("no GPU backend enabled")
}
