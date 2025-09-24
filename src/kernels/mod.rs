/// ## Declare a GPU kernel (CUDA & Metal) and a rust wrapper
/// Usage:
/// ```
/// declare_kernel!("crossfade", UserParams);
/// ```
/// ## Will generate:
/// - const CROSSFADE_SHADER
/// - const KERNEL_ENTRY_POINT
/// - pub unsafe fn crossfade<UP>(...)
#[macro_export]
macro_rules! declare_kernel {
    ($name:ident, $user_params_ty:ty) => {
        const SHADER_SRC: &str = $crate::include_shader!($name);
        const KERNEL_ENTRY_POINT: &str = stringify!($name);

        pub unsafe fn $name(
            config: &$crate::types::Configuration,
            user_params: $user_params_ty,
        ) -> Result<(), &'static str> {
            $crate::backends::dispatch_kernel::<$user_params_ty>(
                config,
                user_params,
                SHADER_SRC,
                KERNEL_ENTRY_POINT,
            )
        }
    };
}
