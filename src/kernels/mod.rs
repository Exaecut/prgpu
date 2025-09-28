
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
        $crate::paste::paste! {
            #[allow(non_upper_case_globals)]
            const [<$name:upper _SHADER_SRC>]: &str = $crate::include_shader!($name);
        }

        $crate::paste::paste! {
            #[allow(non_upper_case_globals)]
            const [<$name:upper _KERNEL_ENTRY_POINT>]: &str = stringify!($name);
        }

        $crate::paste::paste! {
            pub unsafe fn $name(
                config: &$crate::types::Configuration,
                user_params: $user_params_ty,
            ) -> Result<(), &'static str> {
                $crate::backends::dispatch_kernel::<$user_params_ty>(
                    config,
                    user_params,
                    [<$name:upper _SHADER_SRC>],
                    [<$name:upper _KERNEL_ENTRY_POINT>],
                )
            }
        }
    };
}
