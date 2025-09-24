use std::ffi::{CStr, CString};

use after_effects::log;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use std::os::raw::c_void;
use std::time::Instant;

pub unsafe fn nsstring_utf8(s: &str) -> *mut Object {
    let c = CString::new(s).unwrap();
    let ns: *mut Object = msg_send![class!(NSString), stringWithUTF8String: c.as_ptr()];
    ns
}

pub unsafe fn log_buffer_info(tag: &str, raw: *mut core::ffi::c_void) {
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    if raw.is_null() {
        after_effects::log::error!("[metal] {tag}: null");
        return;
    }
    let obj = raw as *mut Object;
    let length: u64 = msg_send![obj, length];

    // storage mode: 0 Shared, 1 Managed, 2 Private, 3 Memoryless
    let storage_mode: u64 = msg_send![obj, storageMode];
    // null if storageMode == Private
    let contents: *mut core::ffi::c_void = msg_send![obj, contents];
    after_effects::log::info!(
        "[metal] {tag}: MTLBuffer={raw:?}, length={length}, storageMode={storage_mode}, contents={contents:?}"
    );
}

pub unsafe fn ns_error(err: *mut Object) -> Option<String> {
    if err.is_null() {
        return None;
    }

    // Domain (NSString*)
    let domain: *mut Object = msg_send![err, domain];
    let domain_c: *const std::os::raw::c_char = msg_send![domain, UTF8String];
    let domain_str = if !domain_c.is_null() {
        unsafe { CStr::from_ptr(domain_c).to_string_lossy().into_owned() }
    } else {
        "<unknown-domain>".into()
    };

    // Code (NSInteger)
    let code: i64 = msg_send![err, code];

    // Localized Description
    let desc: *mut Object = msg_send![err, localizedDescription];
    let desc_c: *const std::os::raw::c_char = msg_send![desc, UTF8String];
    let desc_str = if !desc_c.is_null() {
        unsafe { CStr::from_ptr(desc_c).to_string_lossy().into_owned() }
    } else {
        "<no-description>".into()
    };

    // Localized Failure Reason (often contains compiler error details)
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

    // Localized Recovery Suggestion (sometimes used too)
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
pub use pipeline::{Pipelines, cleanup, get_pso_pair, hot_reload};

use crate::{Configuration, TransitionParams};

pub mod buffer;

pub fn run<UP>(
    config: &Configuration,
    user_params: UP,
    shader_src: &'static str,
    entry: &'static str,
) -> Result<(), &'static str> {
    use objc::rc::autoreleasepool;
    return autoreleasepool(|| {
        use objc::{msg_send, runtime::Object, sel, sel_impl};

        if config.device_handle.is_null() || config.command_queue_handle.is_null() {
            log::error!("Device or command queue handle is null");
            return Err("Invalid device or command queue handle");
        }
        if config.outgoing_data.is_null()
            || config.incoming_data.is_null()
            || config.dest_data.is_null()
        {
            log::error!("[Metal] one of buffers is null");
            return Err("null buffers");
        }

        let device = config.device_handle as *mut Object;
        let queue = config.command_queue_handle as *mut Object;

        let (pso_f32, pso_f16) =
            unsafe { crate::gpu::pipeline::get_pso_pair(device, shader_src, entry) }?;
        let pipeline: *mut Object = if config.is16f { pso_f16 } else { pso_f32 };
        if pipeline.is_null() {
            log::error!("[Metal] Failed to create compute pipeline state");
            return Err("Failed to create Metal compute pipeline state");
        }

        let transition_params = TransitionParams {
            out_pitch: config.outgoing_pitch_px as u32,
            in_pitch: config.incoming_pitch_px as u32,
            dest_pitch: config.dest_pitch_px as u32,
            width: config.width,
            height: config.height,
            progress: config.progress,
        };

        let params_len = std::mem::size_of::<TransitionParams>();
        let transition_params_buffer: *mut Object = unsafe {
            msg_send![
                device,
                newBufferWithBytes: &transition_params as *const _ as *const c_void
                length: params_len
                options: 0u64
            ]
        };

        if transition_params_buffer.is_null() {
            log::error!("[Metal] Failed to create params buffer");
            return Err("Failed to create Metal params buffer");
        }

        let user_params_len = std::mem::size_of::<UP>();

        #[cfg(debug_assertions)]
        log::info!("User params buffer size: {}", user_params_len);

        let user_params_buffer: *mut Object = unsafe {
            msg_send![
                device,
                newBufferWithBytes: &user_params as *const _ as *const c_void
                length: user_params_len
                options: 0u64
            ]
        };

        if user_params_buffer.is_null() {
            log::error!("[Metal] Failed to create user params buffer");
            return Err("Failed to create Metal user params buffer");
        }

        let cmd: *mut Object = unsafe { msg_send![queue, commandBuffer] };
        if cmd.is_null() {
            log::error!("[Metal] Failed to create command buffer");
            return Err("Failed to create Metal command buffer");
        }
        let enc: *mut Object = unsafe { msg_send![cmd, computeCommandEncoder] };
        if enc.is_null() {
            log::error!("[Metal] Failed to create command encoder");
            return Err("Failed to create Metal command encoder");
        }

        unsafe {
            let _: () = msg_send![enc, setComputePipelineState: pipeline];
            let _: () = msg_send![enc, setBuffer: config.outgoing_data as *mut Object  offset: 0  atIndex: 0];
            let _: () = msg_send![enc, setBuffer: config.incoming_data as *mut Object  offset: 0  atIndex: 1];
            let _: () = msg_send![enc, setBuffer: config.dest_data as *mut Object      offset: 0  atIndex: 2];
            let _: () = msg_send![enc, setBuffer: transition_params_buffer             offset: 0  atIndex: 3];
            let _: () = msg_send![enc, setBuffer: user_params_buffer                   offset: 0  atIndex: 4];
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

        // GPU timing available after execution
        let gpu_start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let gpu_end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        let gpu_ms = (gpu_end - gpu_start) * 1000.0;
        let cpu_elapsed = cpu_start.elapsed();

        #[cfg(debug_assertions)]
        log::info!(
            "[Metal] kernel `{}` took {:.3} ms (GPU), {:?} (CPU wall-time)",
            entry,
            gpu_ms,
            cpu_elapsed
        );

        Ok(())
    });
}
