use std::ffi::{CStr, CString};

use after_effects::log;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use std::os::raw::c_void;
use std::time::Instant;

/// Converts a Rust string slice into an Objective-C NSString object.
///
/// Returns an autoreleased NSString. Only valid within a surrounding `autoreleasepool`.
///
/// # Safety
/// The input string must not contain interior null bytes.
pub unsafe fn nsstring_utf8(s: &str) -> *mut Object {
    let c = CString::new(s).unwrap();
    let ns: *mut Object = msg_send![class!(NSString), stringWithUTF8String: c.as_ptr()];
    ns
}

/// # Safety
/// `raw` must be a valid MTLBuffer pointer or null.
pub unsafe fn log_buffer_info(tag: &str, raw: *mut core::ffi::c_void) {
    if raw.is_null() {
        log::error!("[metal] {tag}: null");
        return;
    }
    let obj = raw as *mut Object;
    let length: u64 = msg_send![obj, length];
    let storage_mode: u64 = msg_send![obj, storageMode];
    let contents: *mut core::ffi::c_void = msg_send![obj, contents];
    log::info!(
        "[metal] {tag}: MTLBuffer={raw:?}, length={length}, storageMode={storage_mode}, contents={contents:?}"
    );
}

/// Extracts a readable error string from an NSError pointer.
///
/// # Safety
/// `err` must be a valid NSError pointer or null.
pub unsafe fn ns_error(err: *mut Object) -> Option<String> {
    if err.is_null() {
        return None;
    }

    let domain: *mut Object = msg_send![err, domain];
    let domain_c: *const std::os::raw::c_char = msg_send![domain, UTF8String];
    let domain_str = if !domain_c.is_null() {
        unsafe { CStr::from_ptr(domain_c).to_string_lossy().into_owned() }
    } else {
        "<unknown-domain>".into()
    };

    let code: i64 = msg_send![err, code];

    let desc: *mut Object = msg_send![err, localizedDescription];
    let desc_c: *const std::os::raw::c_char = msg_send![desc, UTF8String];
    let desc_str = if !desc_c.is_null() {
        unsafe { CStr::from_ptr(desc_c).to_string_lossy().into_owned() }
    } else {
        "<no-description>".into()
    };

    let fail: *mut Object = msg_send![err, localizedFailureReason];
    let fail_c: *const std::os::raw::c_char = if fail.is_null() {
        std::ptr::null()
    } else {
        msg_send![fail, UTF8String]
    };
    let fail_str = if !fail_c.is_null() {
        unsafe { CStr::from_ptr(fail_c).to_string_lossy().into_owned() }
    } else {
        String::new()
    };

    let sugg: *mut Object = msg_send![err, localizedRecoverySuggestion];
    let sugg_c: *const std::os::raw::c_char = if sugg.is_null() {
        std::ptr::null()
    } else {
        msg_send![sugg, UTF8String]
    };
    let sugg_str = if !sugg_c.is_null() {
        unsafe { CStr::from_ptr(sugg_c).to_string_lossy().into_owned() }
    } else {
        String::new()
    };

    let mut msg = format!("{domain_str} ({code}): {desc_str}");
    if !fail_str.is_empty() {
        msg.push_str(&format!("\nFailureReason: {fail_str}"));
    }
    if !sugg_str.is_empty() {
        msg.push_str(&format!("\nSuggestion: {sugg_str}"));
    }

    Some(msg)
}

pub mod pipeline;

use crate::{Configuration, FrameParams};

pub mod buffer;

pub fn run<UP>(
    config: &Configuration,
    user_params: UP,
    shader_src: &'static str,
    entry: &'static str,
) -> Result<(), &'static str> {
    use objc::rc::autoreleasepool;
    autoreleasepool(|| {
        if config.device_handle.is_null() || config.command_queue_handle.is_null() {
            log::error!("[Metal] device or command queue handle is null");
            return Err("Invalid device or command queue handle");
        }
        if config.dest_data.is_null() {
            log::error!("[Metal] dest_data is null");
            return Err("null dest buffer");
        }

        let has_outgoing = config
            .outgoing_data
            .map_or(false, |p| !p.is_null());
        let has_incoming = config
            .incoming_data
            .map_or(false, |p| !p.is_null());

        if !has_outgoing && !has_incoming {
            log::error!("[Metal] both outgoing and incoming are null/missing");
            return Err("no input buffers");
        }

        let device = config.device_handle as *mut Object;
        let queue = config.command_queue_handle as *mut Object;

        let (pso_f32, pso_f16) =
            unsafe { crate::gpu::pipeline::get_pso_pair(device, shader_src, entry) }?;
        let pipeline: *mut Object = if config.is16f { pso_f16 } else { pso_f32 };
        if pipeline.is_null() {
            log::error!("[Metal] pipeline state is null");
            return Err("null pipeline state");
        }

        let frame_params = FrameParams {
            out_pitch: config.outgoing_pitch_px as u32,
            in_pitch: config.incoming_pitch_px as u32,
            dest_pitch: config.dest_pitch_px as u32,
            width: config.width,
            height: config.height,
            progress: config.progress,
        };

        let outgoing_ptr = config.outgoing_data.unwrap_or(std::ptr::null_mut());
        let incoming_ptr = config.incoming_data.unwrap_or(std::ptr::null_mut());

        // Constant buffers: Shared storage is correct for single-frame CPU→GPU upload.
        // +1 retained — must release after use.
        let frame_params_buffer: *mut Object = unsafe {
            msg_send![
                device,
                newBufferWithBytes: &frame_params as *const _ as *const c_void
                length: std::mem::size_of::<FrameParams>()
                options: 0u64
            ]
        };
        if frame_params_buffer.is_null() {
            log::error!("[Metal] failed to create params buffer");
            return Err("params buffer allocation failed");
        }

        let user_params_buffer: *mut Object = unsafe {
            msg_send![
                device,
                newBufferWithBytes: &user_params as *const _ as *const c_void
                length: std::mem::size_of::<UP>()
                options: 0u64
            ]
        };
        if user_params_buffer.is_null() {
            unsafe { let _: () = msg_send![frame_params_buffer, release]; }
            log::error!("[Metal] failed to create user params buffer");
            return Err("user params buffer allocation failed");
        }

        let cmd: *mut Object = unsafe { msg_send![queue, commandBuffer] };
        if cmd.is_null() {
            unsafe {
                let _: () = msg_send![frame_params_buffer, release];
                let _: () = msg_send![user_params_buffer, release];
            }
            log::error!("[Metal] failed to create command buffer");
            return Err("command buffer creation failed");
        }

        let enc: *mut Object = unsafe { msg_send![cmd, computeCommandEncoder] };
        if enc.is_null() {
            unsafe {
                let _: () = msg_send![frame_params_buffer, release];
                let _: () = msg_send![user_params_buffer, release];
            }
            log::error!("[Metal] failed to create compute encoder");
            return Err("compute encoder creation failed");
        }

        unsafe {
            let _: () = msg_send![enc, setComputePipelineState: pipeline];
            let _: () = msg_send![enc, setBuffer: outgoing_ptr as *mut Object   offset: 0usize  atIndex: 0usize];
            let _: () = msg_send![enc, setBuffer: incoming_ptr as *mut Object   offset: 0usize  atIndex: 1usize];
            let _: () = msg_send![enc, setBuffer: config.dest_data as *mut Object offset: 0usize atIndex: 2usize];
            let _: () = msg_send![enc, setBuffer: frame_params_buffer       offset: 0usize  atIndex: 3usize];
            let _: () = msg_send![enc, setBuffer: user_params_buffer             offset: 0usize  atIndex: 4usize];
        }

        let tew: usize = unsafe { msg_send![pipeline, threadExecutionWidth] };
        let max_threads: usize = unsafe { msg_send![pipeline, maxTotalThreadsPerThreadgroup] };
        let tg_w = tew.max(1);
        let tg_h = (max_threads / tg_w).clamp(1, 16);
        let groups_x = (config.width as usize).div_ceil(tg_w);
        let groups_y = (config.height as usize).div_ceil(tg_h);

        let tg = crate::types::MTLSize {
            width: groups_x,
            height: groups_y,
            depth: 1,
        };
        let tp = crate::types::MTLSize {
            width: tg_w,
            height: tg_h,
            depth: 1,
        };

        unsafe {
            let _: () = msg_send![enc, dispatchThreadgroups: tg threadsPerThreadgroup: tp];
            let _: () = msg_send![enc, endEncoding];
        }

        let cpu_start = Instant::now();

        unsafe {
            let _: () = msg_send![cmd, commit];
            let _: () = msg_send![cmd, waitUntilCompleted];
        }

        // Check command buffer error status
        let status: u64 = unsafe { msg_send![cmd, status] };
        if status == 5 {
            // MTLCommandBufferStatusError = 5
            let error: *mut Object = unsafe { msg_send![cmd, error] };
            if let Some(msg) = unsafe { ns_error(error) } {
                log::error!("[Metal] command buffer error: {msg}");
            }
            unsafe {
                let _: () = msg_send![frame_params_buffer, release];
                let _: () = msg_send![user_params_buffer, release];
            }
            return Err("GPU execution error");
        }

        let gpu_start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let gpu_end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        let gpu_ms = (gpu_end - gpu_start) * 1000.0;
        let cpu_elapsed = cpu_start.elapsed();

        #[cfg(debug_assertions)]
        log::info!(
            "[Metal] kernel `{entry}` took {gpu_ms:.3} ms (GPU), {cpu_elapsed:?} (CPU wall-time)"
        );

        // Release per-frame constant buffers (+1 retained from newBufferWithBytes)
        unsafe {
            let _: () = msg_send![frame_params_buffer, release];
            let _: () = msg_send![user_params_buffer, release];
        }

        Ok(())
    })
}
