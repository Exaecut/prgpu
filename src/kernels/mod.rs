/// Declares a GPU kernel (CUDA & Metal) + CPU fallback dispatch.
///
/// Usage: `declare_kernel!(vignette, VignetteParams);`
///
/// Generates:
/// - `const VIGNETTE_SHADER_SRC` (embedded PTX)
/// - `const VIGNETTE_KERNEL_ENTRY_POINT`
/// - `pub unsafe fn vignette(config, user_params)` (GPU dispatch)
/// - `pub unsafe fn vignette_cpu(config, user_params)` (CPU fallback dispatch)
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

        $crate::paste::paste! {
            unsafe extern "C" {
                fn [<$name _cpu_dispatch>](
                    buffers: *const *const std::ffi::c_void,
                    transition_params: *const std::ffi::c_void,
                    user_params: *const std::ffi::c_void,
                );
            }

            pub unsafe fn [<$name _cpu>](
                config: &$crate::types::Configuration,
                user_params: $user_params_ty,
            ) -> Result<(), &'static str> {
                let buffers: [*const std::ffi::c_void; 3] = [
                    config.outgoing_data.unwrap_or(std::ptr::null_mut()) as *const std::ffi::c_void,
                    config.incoming_data.unwrap_or(std::ptr::null_mut()) as *const std::ffi::c_void,
                    config.dest_data as *const std::ffi::c_void,
                ];
                let tp = $crate::types::TransitionParams {
                    out_pitch: config.outgoing_pitch_px as u32,
                    in_pitch: config.incoming_pitch_px as u32,
                    dest_pitch: config.dest_pitch_px as u32,
                    width: config.width,
                    height: config.height,
                    progress: config.progress,
                };
                unsafe {
                    [<$name _cpu_dispatch>](
                        buffers.as_ptr(),
                        &tp as *const _ as *const std::ffi::c_void,
                        &user_params as *const _ as *const std::ffi::c_void,
                    );
                }
                Ok(())
            }
        }
    };
}
