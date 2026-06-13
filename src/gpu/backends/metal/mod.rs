use std::ffi::{CStr, CString};

use after_effects::log;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use std::os::raw::c_void;
use std::time::{Duration, Instant};

pub unsafe fn nsstring_utf8(s: &str) -> *mut Object {
	let c = CString::new(s).unwrap();
	let ns: *mut Object = msg_send![class!(NSString), stringWithUTF8String: c.as_ptr()];
	ns
}

pub unsafe fn log_buffer_info(tag: &str, raw: *mut core::ffi::c_void) {
	if raw.is_null() {
		log::error!("[metal] {tag}: null");
		return;
	}
	let obj = raw as *mut Object;
	let length: u64 = msg_send![obj, length];
	let storage_mode: u64 = msg_send![obj, storageMode];
	let contents: *mut core::ffi::c_void = msg_send![obj, contents];
	log::info!("[metal] {tag}: MTLBuffer={raw:?}, length={length}, storageMode={storage_mode}, contents={contents:?}");
}

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
	let fail_c: *const std::os::raw::c_char = if fail.is_null() { std::ptr::null() } else { msg_send![fail, UTF8String] };
	let fail_str = if !fail_c.is_null() {
		unsafe { CStr::from_ptr(fail_c).to_string_lossy().into_owned() }
	} else {
		String::new()
	};

	let sugg: *mut Object = msg_send![err, localizedRecoverySuggestion];
	let sugg_c: *const std::os::raw::c_char = if sugg.is_null() { std::ptr::null() } else { msg_send![sugg, UTF8String] };
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

pub mod buffer;
pub mod fence;
pub mod frame_scope;
pub mod pipeline;

use crate::types::{Configuration, FrameParams};

// setBytes is only valid for argument data up to 4 KB.
const SET_BYTES_LIMIT: usize = 4096;

pub fn run<UP>(config: &Configuration, user_params: UP, shader_src: &[u8], entry: &'static str) -> Result<(), &'static str> {
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

		let has_outgoing = config.outgoing_data.map_or(false, |p| !p.is_null());
		let has_incoming = config.incoming_data.map_or(false, |p| !p.is_null());

		if !has_outgoing && !has_incoming {
			log::error!("[Metal] both outgoing and incoming are null/missing");
			return Err("no input buffers");
		}

		let device = config.device_handle as *mut Object;
		let queue = config.command_queue_handle as *mut Object;

		let pipeline = unsafe { crate::gpu::pipeline::load_kernel(device, shader_src, entry) }?;
		if pipeline.is_null() {
			log::error!("[Metal] pipeline state is null");
			return Err("null pipeline state");
		}

		// out_desc/in_desc describe SOURCE buffers (may be downsampled); dst_desc + width/height drive the dispatch grid.
		let frame_params = FrameParams::from_config(config);

		let outgoing_ptr = config.outgoing_data.unwrap_or(std::ptr::null_mut());
		let incoming_ptr = config.incoming_data.unwrap_or(std::ptr::null_mut());

		// Params go through setBytes (Metal's by-value constant path): no
		// MTLBuffer alloc/release per pass. Valid only below 4 KB.
		let frame_params_size = std::mem::size_of::<FrameParams>();
		let user_param_size = std::mem::size_of::<UP>();
		debug_assert!(frame_params_size <= SET_BYTES_LIMIT && user_param_size <= SET_BYTES_LIMIT);

		#[cfg(debug_assertions)]
		log::debug!(
			"[Metal] '{entry}' bufs: dispatch={}x{} dst_pitch_px={} | outgoing={}x{} out_pitch_px={} mip_levels={} outDesc.mipCount={} | dstDesc={}x{} dstDesc.pitch={} | outgoing_ptr={:?} incoming_ptr={:?} dst_ptr={:?}",
			config.width,
			config.height,
			config.dest_pitch_px,
			config.outgoing_width,
			config.outgoing_height,
			config.outgoing_pitch_px,
			config.outgoing_mip_levels,
			frame_params.out_desc.mip_level_count,
			frame_params.dst_desc.width,
			frame_params.dst_desc.height,
			frame_params.dst_desc.pitch_bytes,
			outgoing_ptr,
			incoming_ptr,
			config.dest_data,
		);

		// Threadgroup geometry is invariant across retries; derive it once.
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

		// Inside a frame scope, encode into the frame's command buffer and let
		// the adapter commit + wait once; the watchdog retry lives there too.
		if frame_scope::is_active() {
			let cmd = frame_scope::command_buffer();
			let enc: *mut Object = unsafe { msg_send![cmd, computeCommandEncoder] };
			if enc.is_null() {
				log::error!("[Metal] failed to create compute encoder");
				return Err("compute encoder creation failed");
			}
			unsafe {
				encode_pass(enc, pipeline, outgoing_ptr, incoming_ptr, config.dest_data, &frame_params, &user_params, tg, tp);
			}
			frame_scope::note_pass();
			return Ok(());
		}

		// Standalone dispatch (tests, single-pass callers): own command buffer,
		// commit, single wait. macOS Metal's GPU watchdog
		// (kIOGPUCommandBufferCallbackError / "Impacting Interactivity") aborts
		// command buffers that exceed the OS budget; first dispatches of a heavy
		// kernel typically trip it because pipeline JIT, cold caches, and
		// Premiere's concurrent decode/UI all land at once. Retry once with a
		// cool-down; non-watchdog errors still propagate.
		const MAX_ATTEMPTS: u32 = 2;
		let mut attempt: u32 = 0;
		let gpu_ms = loop {
			attempt += 1;

			let cmd: *mut Object = unsafe { msg_send![queue, commandBuffer] };
			if cmd.is_null() {
				log::error!("[Metal] failed to create command buffer");
				return Err("command buffer creation failed");
			}

			let enc: *mut Object = unsafe { msg_send![cmd, computeCommandEncoder] };
			if enc.is_null() {
				log::error!("[Metal] failed to create compute encoder");
				return Err("compute encoder creation failed");
			}

			unsafe {
				encode_pass(enc, pipeline, outgoing_ptr, incoming_ptr, config.dest_data, &frame_params, &user_params, tg, tp);
			}

			#[cfg(debug_assertions)]
			let cpu_start = Instant::now();

			unsafe {
				let _: () = msg_send![cmd, commit];
				let _: () = msg_send![cmd, waitUntilCompleted];
			}

			let status: u64 = unsafe { msg_send![cmd, status] };
			if status == 5 {
				let error: *mut Object = unsafe { msg_send![cmd, error] };
				let msg = unsafe { ns_error(error) };
				let is_watchdog = msg
					.as_ref()
					.is_some_and(|m| m.contains("Impacting Interactivity") || m.contains("kIOGPUCommandBufferCallbackError"));

				if is_watchdog && attempt < MAX_ATTEMPTS {
					log::warn!(
						"[Metal] '{entry}' hit GPU watchdog (attempt {attempt}/{MAX_ATTEMPTS}) — cooling down 50ms and retrying"
					);
					std::thread::sleep(Duration::from_millis(50));
					continue;
				}

				if let Some(m) = msg {
					log::error!("[Metal] command buffer error: {m}");
				}
				return Err("GPU execution error");
			}

			if attempt > 1 {
				log::info!("[Metal] '{entry}' recovered after watchdog retry (attempt {attempt})");
			}

			let gpu_start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
			let gpu_end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
			let gpu_ms = (gpu_end - gpu_start) * 1000.0;

			#[cfg(debug_assertions)]
			{
				let cpu_elapsed = cpu_start.elapsed();
				let generation = config.render_generation;
				log::info!("[Metal] `{entry}` gen={generation}: gpu={gpu_ms:.3}ms, cpu={cpu_elapsed:?}");
			}

			break gpu_ms;
		};

		crate::timing::record(entry, crate::types::Backend::Metal, (gpu_ms * 1_000_000.0) as u64);

		Ok(())
	})
}

/// Encode one compute pass: pipeline, the 5-slot buffer convention
/// (outgoing / incoming / dst / frame / params), dispatch, end encoding.
/// Params bind via setBytes — no MTLBuffer alloc.
///
/// # Safety: `enc` and `pipeline` valid; buffer pointers follow the
/// `Configuration` lifetime contract.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_pass<UP>(
	enc: *mut Object,
	pipeline: *mut Object,
	outgoing: *mut c_void,
	incoming: *mut c_void,
	dest: *mut c_void,
	frame_params: &FrameParams,
	user_params: &UP,
	tg: crate::types::MTLSize,
	tp: crate::types::MTLSize,
) {
	unsafe {
		let _: () = msg_send![enc, setComputePipelineState: pipeline];
		let _: () = msg_send![enc, setBuffer: outgoing as *mut Object offset: 0usize atIndex: 0usize];
		let _: () = msg_send![enc, setBuffer: incoming as *mut Object offset: 0usize atIndex: 1usize];
		let _: () = msg_send![enc, setBuffer: dest as *mut Object offset: 0usize atIndex: 2usize];
		let _: () = msg_send![enc, setBytes: frame_params as *const _ as *const c_void length: std::mem::size_of::<FrameParams>() atIndex: 3usize];
		let _: () = msg_send![enc, setBytes: user_params as *const _ as *const c_void length: std::mem::size_of::<UP>() atIndex: 4usize];
		let _: () = msg_send![enc, dispatchThreadgroups: tg threadsPerThreadgroup: tp];
		let _: () = msg_send![enc, endEncoding];
	}
}
