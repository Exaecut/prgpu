use std::ffi::{CStr, CString, c_char};

use bytemuck::{Pod, Zeroable};

pub const VEKL_LOG_LEVEL_TRACE: u32 = 0;
pub const VEKL_LOG_LEVEL_DEBUG: u32 = 1;
pub const VEKL_LOG_LEVEL_INFO: u32 = 2;
pub const VEKL_LOG_LEVEL_WARN: u32 = 3;
pub const VEKL_LOG_LEVEL_ERROR: u32 = 4;

pub const VEKL_LOG_CAPACITY: usize = 1024;
pub const VEKL_LOG_CHANNEL_CAPACITY: usize = 32;
pub const VEKL_LOG_MESSAGE_CAPACITY: usize = 160;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct VeklLogEntry {
	pub committed_sequence: u32,
	pub level: u32,
	pub channel_len: u32,
	pub message_len: u32,
	pub channel: [u8; VEKL_LOG_CHANNEL_CAPACITY],
	pub message: [u8; VEKL_LOG_MESSAGE_CAPACITY],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct VeklLogBuffer {
	pub write_index: u32,
	pub capacity: u32,
	pub overrun_count: u32,
	pub reserved: u32,
	pub entries: [VeklLogEntry; VEKL_LOG_CAPACITY],
}

impl VeklLogBuffer {
	pub fn initialize(&mut self) {
		*self = Self::zeroed();
		self.capacity = VEKL_LOG_CAPACITY as u32;
	}

	pub fn zeroed() -> Self {
		Self {
			write_index: 0,
			capacity: 0,
			overrun_count: 0,
			reserved: 0,
			entries: [VeklLogEntry::zeroed(); VEKL_LOG_CAPACITY],
		}
	}

	pub fn capacity(&self) -> u32 {
		self.capacity.clamp(1, VEKL_LOG_CAPACITY as u32)
	}

	pub fn is_ready_for_host(&self) -> bool {
		self.capacity != 0
	}

	pub fn drain_into_host_log(&self, cursor: &mut GpuLogCursor, kernel_name: &str, backend_name: &str) -> usize {
		drain_gpu_log_buffer(self, cursor, kernel_name, backend_name)
	}

	pub fn as_mut_ptr(&mut self) -> *mut Self {
		self as *mut Self
	}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GpuLogCursor {
	pub next_sequence: u32,
	pub observed_overrun_count: u32,
	pub dropped_entries: u64,
}

#[inline]
fn emit_log(level: u32, message: &str) {
	match level {
		VEKL_LOG_LEVEL_TRACE => after_effects::log::trace!("{message}"),
		VEKL_LOG_LEVEL_DEBUG => after_effects::log::debug!("{message}"),
		VEKL_LOG_LEVEL_INFO => after_effects::log::info!("{message}"),
		VEKL_LOG_LEVEL_WARN => after_effects::log::warn!("{message}"),
		VEKL_LOG_LEVEL_ERROR => after_effects::log::error!("{message}"),
		_ => after_effects::log::info!("{message}"),
	}
}

#[inline]
fn forward_to_host_log(level: u32, message: &str) {
	match CString::new(message) {
		Ok(message) => unsafe {
			host_log(level, message.as_ptr());
		},
		Err(_) => {
			let sanitized = message.replace('\0', "␀");
			emit_log(level, &sanitized);
		}
	}
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn host_log(level: u32, message: *const c_char) {
	if message.is_null() {
		after_effects::log::warn!("[host_log] received a null message pointer");
		return;
	}

	let message = unsafe { CStr::from_ptr(message) };
	let message = message.to_string_lossy();
	emit_log(level, &message);
}

fn decode_lossy(bytes: &[u8], declared_len: u32) -> String {
	let len = declared_len.min(bytes.len() as u32) as usize;
	let bytes = &bytes[..len];
	let bytes = if let Some(end) = bytes.iter().position(|byte| *byte == 0) {
		&bytes[..end]
	} else {
		bytes
	};
	String::from_utf8_lossy(bytes).into_owned()
}

pub fn drain_gpu_log_buffer(buffer: &VeklLogBuffer, cursor: &mut GpuLogCursor, kernel_name: &str, backend_name: &str) -> usize {
	if !buffer.is_ready_for_host() {
		return 0;
	}

	let capacity = buffer.capacity();
	let write_index = buffer.write_index;
	let available = write_index.wrapping_sub(cursor.next_sequence);

	if available > capacity {
		let skipped = available - capacity;
		cursor.dropped_entries += u64::from(skipped);
		cursor.next_sequence = write_index.wrapping_sub(capacity);
		forward_to_host_log(
			VEKL_LOG_LEVEL_WARN,
			&format!(
				"[{kernel_name}({backend_name})/logging] - GPU log buffer overrun; skipped {skipped} entr(y/ies) (device overrun_count={})",
				buffer.overrun_count
			),
		);
	}

	let mut drained = 0usize;
	while cursor.next_sequence != write_index {
		let slot = (cursor.next_sequence % capacity) as usize;
		let expected_sequence = cursor.next_sequence.wrapping_add(1);
		let entry = &buffer.entries[slot];

		if entry.committed_sequence != expected_sequence {
			break;
		}

		let channel = decode_lossy(&entry.channel, entry.channel_len);
		let message = decode_lossy(&entry.message, entry.message_len);
		let channel = if channel.is_empty() { "default" } else { channel.as_str() };

		forward_to_host_log(entry.level, &format!("[{kernel_name}({backend_name})/{channel}] - {message}"));

		cursor.next_sequence = cursor.next_sequence.wrapping_add(1);
		drained += 1;
	}

	cursor.observed_overrun_count = buffer.overrun_count;
	drained
}
