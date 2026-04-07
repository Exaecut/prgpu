use after_effects::log;
use std::ffi::c_void;
use std::ptr::null_mut;

use cudarc::driver::sys as cuda;

pub mod buffer;
pub mod pipeline;
pub mod fence;

use crate::{Configuration, FrameParams};

#[inline]
fn check(res: cuda::CUresult, what: &str) -> Result<(), &'static str> {
    if res == cuda::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut err_str: *const i8 = std::ptr::null();
    unsafe { cuda::cuGetErrorString(res, &mut err_str) };
    let msg = if err_str.is_null() {
        what.to_string()
    } else {
        unsafe {
            std::ffi::CStr::from_ptr(err_str)
                .to_string_lossy()
                .to_string()
        }
    };
    log::error!("[CUDA] {what} failed: {msg}");
    Err("CUDA error")
}

#[inline]
#[allow(dead_code)]
unsafe fn compute_capability(dev: cuda::CUdevice) -> Result<(i32, i32), &'static str> {
    let mut major = 0;
    let mut minor = 0;
    check(
        unsafe {
            cuda::cuDeviceGetAttribute(
                &mut major,
                cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            )
        },
        "cuDeviceGetAttribute(MAJOR)",
    )?;
    check(
        unsafe {
            cuda::cuDeviceGetAttribute(
                &mut minor,
                cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            )
        },
        "cuDeviceGetAttribute(MINOR)",
    )?;
    Ok((major, minor))
}

/// Launches a CUDA kernel on the given stream. Does NOT synchronize.
///
/// # Safety
/// - `ctx`, `stream`, `func` must be valid CUDA handles.
/// - `params` must point to valid device memory matching the kernel signature.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch(
    ctx: *mut c_void,
    stream: *mut c_void,
    func: cuda::CUfunction,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    block_y: u32,
    params: &mut [*mut c_void],
) -> Result<(), &'static str> {
    if ctx.is_null() || stream.is_null() || func.is_null() {
        log::error!("[CUDA] dispatch - null handle");
        return Err("null handle");
    }
    check(
        unsafe { cuda::cuCtxSetCurrent(ctx as cuda::CUcontext) },
        "cuCtxSetCurrent",
    )?;
    check(
        unsafe {
            cuda::cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                block_x,
                block_y,
                1,
                0,
                stream as cuda::CUstream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "cuLaunchKernel",
    )?;
    Ok(())
}

/// # Safety
/// `ptr` must be a valid CUDA device pointer or null.
pub unsafe fn log_device_ptr_info(tag: &str, ptr: *mut c_void) {
    if ptr.is_null() {
        log::error!("[cuda] {tag}: null");
        return;
    }
    let mut mem_type: i32 = 0;
    let _ = unsafe {
        cuda::cuPointerGetAttribute(
            &mut mem_type as *mut _ as *mut c_void,
            cuda::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            ptr as u64,
        )
    };
    log::info!("[cuda] {tag}: CUdeviceptr={ptr:?}, memory_type={mem_type}");
}

pub fn run<UP>(
    config: &Configuration,
    user_params: UP,
    shader_src: &'static str,
    shader_src_f16: &'static str,
    entry: &'static str,
) -> Result<(), &'static str> {
    use crate::gpu;

    if config.context_handle.is_none() || config.command_queue_handle.is_null() {
        log::error!("[CUDA] invalid handles");
        return Err("Invalid CUDA handles");
    }
    if config.dest_data.is_null() {
        log::error!("[CUDA] dest_data can't be null");
        return Err("null buffers");
    }

    let ctx = config.context_handle.unwrap();

    let (func_f32, func_f16) =
        unsafe { gpu::pipeline::load_kernel(ctx as _, shader_src, shader_src_f16, entry) }
            .map_err(|e| {
                log::error!("[CUDA] {e}");
                "kernel load failed"
            })?;
    let func = if config.is16f { func_f16 } else { func_f32 };

    let outgoing_data = config.outgoing_data.unwrap_or(null_mut());
    let incoming_data = config.incoming_data.unwrap_or(null_mut());

    let mut d_outgoing = outgoing_data as u64;
    let mut d_incoming = incoming_data as u64;
    let mut d_dest = config.dest_data as u64;

    let mut p = FrameParams {
        out_pitch: config.outgoing_pitch_px as u32,
        in_pitch: config.incoming_pitch_px as u32,
        dest_pitch: config.dest_pitch_px as u32,
        width: config.width,
        height: config.height,
        progress: config.progress,
    };
    let mut u = user_params;

    let mut params: [*mut c_void; 5] = [
        &mut d_outgoing as *mut _ as *mut c_void,
        &mut d_incoming as *mut _ as *mut c_void,
        &mut d_dest as *mut _ as *mut c_void,
        &mut p as *mut _ as *mut c_void,
        &mut u as *mut _ as *mut c_void,
    ];

    let block_x: u32 = 16;
    let block_y: u32 = 16;
    let grid_x: u32 = config.width.div_ceil(block_x);
    let grid_y: u32 = config.height.div_ceil(block_y);

    unsafe {
        dispatch(
            ctx,
            config.command_queue_handle,
            func,
            grid_x,
            grid_y,
            block_x,
            block_y,
            &mut params,
        )?;
    }

    Ok(())
}
