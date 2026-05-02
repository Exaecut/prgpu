//! Per-dispatch diagnostics for the CPU render path.
//!
//! Emits a single structured log line at the end of each `render_cpu` /
//! `render_cpu_direct` call so we can attribute latency to setup, rayon body,
//! and concurrency — independently from the `timing` aggregate.
//!
//! The log line is **throttled** (default: every 60th dispatch) so emitting
//! the diagnostic does not itself become a source of `OutputDebugStringW`
//! contention. Peak-case numbers (`max`, `min`) continue to be captured by
//! the `timing` aggregate regardless of the throttle.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static CONCURRENT_RENDERS: AtomicUsize = AtomicUsize::new(0);

/// How many dispatches occur between emitted `[dispatch]` log lines.
/// `0` disables throttling. Default: 60.
static LOG_INTERVAL: AtomicU64 = AtomicU64::new(60);
static LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configure how often dispatch lines are emitted. `0` = every call,
/// `N` = roughly 1 line every N calls. Default: 60.
pub fn set_log_interval(interval: u64) {
	LOG_INTERVAL.store(interval, Ordering::Relaxed);
}

/// RAII guard tracking in-flight CPU render dispatches. Increments a global
/// atomic on construction and decrements on drop so we can observe how many
/// frames Premiere is rendering in parallel at the moment our dispatch runs.
pub struct DispatchGuard {
	snapshot_at_entry: usize,
}

impl DispatchGuard {
	#[inline]
	pub fn enter() -> Self {
		// fetch_add returns the *previous* value; +1 is our own entry.
		let previous = CONCURRENT_RENDERS.fetch_add(1, Ordering::Relaxed);
		Self {
			snapshot_at_entry: previous + 1,
		}
	}

	#[inline]
	pub fn concurrent_at_entry(&self) -> usize {
		self.snapshot_at_entry
	}
}

impl Drop for DispatchGuard {
	#[inline]
	fn drop(&mut self) {
		CONCURRENT_RENDERS.fetch_sub(1, Ordering::Relaxed);
	}
}

/// Peek the current number of in-flight CPU dispatches without entering one.
#[inline]
pub fn concurrent_renders() -> usize {
	CONCURRENT_RENDERS.load(Ordering::Relaxed)
}

/// Emit one diagnostic line summarizing a single CPU dispatch, throttled by
/// [`set_log_interval`]. The atomic counter increments unconditionally so
/// even silent dispatches contribute to the rotation.
#[inline]
pub fn log_dispatch(
	kernel: &str,
	path: DispatchPath,
	width: u32,
	height: u32,
	chunk_rows: u32,
	setup_ns: u64,
	rayon_ns: u64,
	concurrent_at_entry: usize,
) {
	let interval = LOG_INTERVAL.load(Ordering::Relaxed);
	if interval != 0 {
		let prev = LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
		if prev % interval != 0 {
			return;
		}
	}

	let pixels = (width as u64) * (height as u64);
	let total_ns = setup_ns + rayon_ns;
	let workers = crate::cpu::pool::worker_count();
	after_effects::log::info!(
		"[{kernel}][dispatch][{path}] w={width} h={height} px={pixels} rows={height} chunk_rows={chunk_rows} setup={setup_us:.1}µs rayon={rayon_us:.1}µs total={total_us:.1}µs concurrent={concurrent_at_entry} workers={workers}",
		path = path.as_str(),
		setup_us = setup_ns as f64 / 1_000.0,
		rayon_us = rayon_ns as f64 / 1_000.0,
		total_us = total_ns as f64 / 1_000.0,
	);
}

/// Which dispatcher variant produced this line.
#[derive(Debug, Clone, Copy)]
pub enum DispatchPath {
	/// Pure rayon (`render_cpu_direct`) — used by Premiere render and benches.
	Direct,
	/// AE `iterate_with` fallback path (After Effects only).
	AeIterate,
	/// Rayon path under `render_cpu` (host has AE shape but falls through
	/// to rayon because sizes mismatch or host is Premiere).
	Rayon,
}

impl DispatchPath {
	#[inline]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Direct => "direct",
			Self::AeIterate => "ae",
			Self::Rayon => "rayon",
		}
	}
}
