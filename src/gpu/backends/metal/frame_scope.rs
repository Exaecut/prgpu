//! Per-frame Metal submission scope.
//!
//! One `MTLCommandBuffer` per frame: every compute pass and blit encodes into
//! it and a single `waitUntilCompleted` runs at the adapter boundary, instead
//! of the per-pass commit + busy-wait + wait that dominated the smoothness gap.
//! The macOS GPU watchdog retry moves to the
//! frame level: [`end`] returns [`ERR_WATCHDOG`] so the adapter can re-run the
//! whole frame once.

use std::cell::Cell;

use after_effects::log;
use objc::{msg_send, runtime::Object, sel, sel_impl};

use crate::types::FrameScopeDesc;

pub const ERR_WATCHDOG: &str = "metal frame watchdog";

#[derive(Clone, Copy)]
struct Scope {
	active: bool,
	cmd: usize,
	passes: u32,
}

impl Scope {
	const fn inactive() -> Self {
		Self { active: false, cmd: 0, passes: 0 }
	}
}

thread_local! {
	static SCOPE: Cell<Scope> = const { Cell::new(Scope::inactive()) };
}

/// Enter the frame scope: create (and retain) the frame's command buffer.
/// No-op when the descriptor carries no Metal queue.
pub fn begin(desc: &FrameScopeDesc) {
	if desc.command_queue_handle.is_null() {
		return;
	}
	let queue = desc.command_queue_handle as *mut Object;
	// Retain inside the pool: the autoreleased command buffer must survive
	// until end(), which may run outside any autoreleasepool.
	let cmd = objc::rc::autoreleasepool(|| {
		let cmd: *mut Object = unsafe { msg_send![queue, commandBuffer] };
		if !cmd.is_null() {
			let _: *mut Object = unsafe { msg_send![cmd, retain] };
		}
		cmd
	});
	if cmd.is_null() {
		log::error!("[Metal/frame] commandBuffer() returned null at frame begin");
		return;
	}
	SCOPE.with(|s| {
		s.set(Scope {
			active: true,
			cmd: cmd as usize,
			passes: 0,
		})
	});
}

/// Commit the frame command buffer and block until it completes. Returns
/// [`ERR_WATCHDOG`] when macOS aborted it (kIOGPUCommandBufferCallbackError /
/// "Impacting Interactivity") so the adapter can retry the whole frame.
pub fn end(desc: &FrameScopeDesc) -> Result<(), &'static str> {
	let scope = SCOPE.with(|s| s.replace(Scope::inactive()));
	if !scope.active {
		return Ok(());
	}
	let cmd = scope.cmd as *mut Object;

	unsafe {
		let _: () = msg_send![cmd, commit];
		let _: () = msg_send![cmd, waitUntilCompleted];
	}

	let status: u64 = unsafe { msg_send![cmd, status] };
	let result = if status == 5 {
		let error: *mut Object = unsafe { msg_send![cmd, error] };
		let msg = unsafe { super::ns_error(error) };
		let is_watchdog = msg
			.as_ref()
			.is_some_and(|m| m.contains("Impacting Interactivity") || m.contains("kIOGPUCommandBufferCallbackError"));
		if let Some(m) = &msg {
			log::error!("[Metal/frame] command buffer error: {m}");
		}
		if is_watchdog { Err(ERR_WATCHDOG) } else { Err("GPU execution error") }
	} else {
		let gpu_start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
		let gpu_end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
		let gpu_ms = (gpu_end - gpu_start) * 1000.0;
		crate::timing::record("frame", crate::types::Backend::Metal, (gpu_ms * 1_000_000.0) as u64);
		log::debug!(
			"[Metal/frame] gen={} cmd_buffers=1 waits=1 passes={} gpu_ms={gpu_ms:.3}",
			desc.render_generation,
			scope.passes
		);
		Ok(())
	};

	unsafe {
		let _: () = msg_send![cmd, release];
	}
	result
}

pub(crate) fn is_active() -> bool {
	SCOPE.with(|s| s.get().active)
}

/// Frame command buffer while the scope is active, else null.
pub(crate) fn command_buffer() -> *mut Object {
	SCOPE.with(|s| s.get().cmd as *mut Object)
}

pub(crate) fn note_pass() {
	SCOPE.with(|s| {
		let mut v = s.get();
		if v.active {
			v.passes += 1;
			s.set(v);
		}
	});
}

/// No-op for API parity with the CUDA frame scope (which owns a device arena).
/// # Safety: no preconditions.
pub unsafe fn cleanup() {}
