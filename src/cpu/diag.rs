//! Per-dispatch diagnostics for the CPU render path.
//!
//! Emits one structured `[dispatch]` line per `render_cpu` / `render_cpu_direct`
//! call, attributing latency to setup, rayon body, and host concurrency. Throttled
//! (default: every 60th dispatch) so the diagnostic itself doesn't become a
//! source of `OutputDebugStringW` contention.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static CONCURRENT_RENDERS: AtomicUsize = AtomicUsize::new(0);

/// Emit one line every Nth dispatch. `0` disables throttling. Default: 60.
static LOG_INTERVAL: AtomicU64 = AtomicU64::new(60);
static LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn set_log_interval(interval: u64) {
	LOG_INTERVAL.store(interval, Ordering::Relaxed);
}

/// RAII guard tracking in-flight CPU dispatches; bumps a global atomic so we can observe how many frames Premiere renders concurrently.
pub struct DispatchGuard {
	snapshot_at_entry: usize,
}

impl DispatchGuard {
	#[inline]
	pub fn enter() -> Self {
		// fetch_add returns the previous value; +1 is our own entry.
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

#[inline]
pub fn concurrent_renders() -> usize {
	CONCURRENT_RENDERS.load(Ordering::Relaxed)
}

/// Emit one diagnostic line, throttled by `set_log_interval`. The counter increments on every call so silent dispatches still rotate.
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

#[derive(Debug, Clone, Copy)]
pub enum DispatchPath {
	/// Pure rayon path (`render_cpu_direct`). Premiere render and benches.
	Direct,
	/// AE `iterate_with` fallback path.
	AeIterate,
	/// Rayon under `render_cpu` (host has AE shape but sizes mismatch, or host is Premiere).
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
