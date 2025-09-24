use after_effects::log;
use parking_lot::Mutex;
use std::{borrow::Cow, collections::HashMap, ffi::c_void, sync::OnceLock};

use cudarc::driver::sys as cu;

#[inline]
fn check(res: cu::CUresult, what: &str) -> Result<(), &'static str> {
    if res == cu::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut err_str: *const i8 = std::ptr::null();
    unsafe { cu::cuGetErrorString(res, &mut err_str) };
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
unsafe fn compute_capability(dev: cu::CUdevice) -> Result<(i32, i32), &'static str> {
    let mut major = 0;
    let mut minor = 0;
    check(
        unsafe {
            cu::cuDeviceGetAttribute(
                &mut major,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            )
        },
        "cuDeviceGetAttribute(MAJOR)",
    )?;
    check(
        unsafe {
            cu::cuDeviceGetAttribute(
                &mut minor,
                cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            )
        },
        "cuDeviceGetAttribute(MINOR)",
    )?;
    Ok((major, minor))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch(
    ctx: *mut c_void,
    stream: *mut c_void,
    func: cu::CUfunction,
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
        unsafe { cu::cuCtxSetCurrent(ctx as cu::CUcontext) },
        "cuCtxSetCurrent",
    )?;
    check(
        unsafe {
            cu::cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                block_x,
                block_y,
                1,
                0,
                stream as cu::CUstream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "cuLaunchKernel",
    )?;

    #[cfg(debug_assertions)]
    {
        check(
            unsafe { cu::cuStreamSynchronize(stream as cu::CUstream) },
            "cuStreamSynchronize",
        )?;
    }
    Ok(())
}

pub unsafe fn log_device_ptr_info(tag: &str, ptr: *mut c_void) {
    if ptr.is_null() {
        log::error!("[cuda] {tag}: null");
        return;
    }
    let mut mem_type: i32 = 0;
    let _ = unsafe {
        cu::cuPointerGetAttribute(
            &mut mem_type as *mut _ as *mut c_void,
            cu::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            ptr as u64,
        )
    };
    log::info!("[cuda] {tag}: CUdeviceptr={ptr:?}, memory_type={mem_type}");
}

pub mod pipeline;
pub use pipeline::*;

pub fn run<UP>(
    config: &Configuration,
    user_params: UP,
    shader_src: &'static str,
    entry: &'static str,
) -> Result<(), &'static str> {
    use crate::gpu;
    use std::time::Instant;

    if config.context_handle.is_none() || config.command_queue_handle.is_null() {
        log::error!("[CUDA] invalid handles");
        return Err("Invalid CUDA handles");
    }
    if config.outgoing_data.is_null()
        || config.incoming_data.is_null()
        || config.dest_data.is_null()
    {
        log::error!("[CUDA] one of buffers is null");
        return Err("null buffers");
    }

    let (func_f32, func_f16) = unsafe {
        gpu::pipeline::get_pso_pair(
            config.context_handle.unwrap() as _,
            shader_src,
            entry,
            config.device_handle,
        )
    }?;
    let func = if config.is16f { func_f16 } else { func_f32 };

    let mut d_outgoing = config.outgoing_data as u64;
    let mut d_incoming = config.incoming_data as u64;
    let mut d_dest = config.dest_data as u64;

    let mut p = super::TransitionParams {
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

    // --- GPU timing with CUDA events ---
    let mut start_event: cuda::CUevent = std::ptr::null_mut();
    let mut end_event: cuda::CUevent = std::ptr::null_mut();
    unsafe {
        cuda::cuEventCreate(&mut start_event, 0);
        cuda::cuEventCreate(&mut end_event, 0);
    }

    let cpu_start = Instant::now();

    unsafe {
        cuda::cuEventRecord(start_event, config.command_queue_handle);
        cuda::dispatch(
            config.context_handle.unwrap(),
            config.command_queue_handle,
            func,
            grid_x,
            grid_y,
            block_x,
            block_y,
            &mut params,
        )?;
        cuda::cuEventRecord(end_event, config.command_queue_handle);
        cuda::cuEventSynchronize(end_event);
    }?;

    let cpu_elapsed = cpu_start.elapsed();

    let mut ms: f32 = 0.0;
    unsafe {
        cuda::cuEventElapsedTime(&mut ms as *mut f32, start_event, end_event);
        cuda::cuEventDestroy(start_event);
        cuda::cuEventDestroy(end_event);
    }

    #[cfg(debug_assertions)]
    log::info!(
        "[CUDA] kernel `{}` took {:.3} ms (GPU), {:?} (CPU wall-time)",
        KERNEL_ENTRY_POINT,
        ms,
        cpu_elapsed
    );

    Ok(())
}
