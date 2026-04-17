//! Per-kernel dispatch timing for CPU and GPU backends.
//!
//! Enabled via `features = ["timing"]` in Cargo.toml.
//! When disabled, all public functions are no-op stubs with zero runtime overhead.

pub use crate::types::Backend;

/// Statistics for a single kernel accumulated across dispatches.
#[derive(Debug, Clone)]
pub struct KernelTiming {
	pub name: &'static str,
	pub backend: Backend,
	pub dispatch_count: u64,
	pub total_ns: u64,
	pub min_ns: u64,
	pub max_ns: u64,
	pub last_ns: u64,
}

impl KernelTiming {
	/// Average time per dispatch in nanoseconds.
	pub fn avg_ns(&self) -> u64 {
		if self.dispatch_count == 0 {
			0
		} else {
			self.total_ns / self.dispatch_count
		}
	}

	/// Average time per dispatch in milliseconds.
	pub fn avg_ms(&self) -> f64 {
		self.avg_ns() as f64 / 1_000_000.0
	}

	/// Minimum dispatch time in milliseconds.
	pub fn min_ms(&self) -> f64 {
		self.min_ns as f64 / 1_000_000.0
	}

	/// Maximum dispatch time in milliseconds.
	pub fn max_ms(&self) -> f64 {
		self.max_ns as f64 / 1_000_000.0
	}

	/// Last dispatch time in milliseconds.
	pub fn last_ms(&self) -> f64 {
		self.last_ns as f64 / 1_000_000.0
	}
}

// ---------------------------------------------------------------------------
// Full implementation when the `timing` feature is active
// ---------------------------------------------------------------------------
#[cfg(feature = "timing")]
mod imp {
	use super::{Backend, KernelTiming};
	use parking_lot::Mutex;
	use std::collections::HashMap;
	use std::sync::atomic::{AtomicBool, Ordering};
	use std::sync::OnceLock;

	static ENABLED: AtomicBool = AtomicBool::new(true);

	struct PerKernelStats {
		backend: Backend,
		dispatch_count: u64,
		total_ns: u64,
		min_ns: u64,
		max_ns: u64,
		last_ns: u64,
	}

	static TIMINGS: OnceLock<Mutex<HashMap<&'static str, PerKernelStats>>> = OnceLock::new();

	fn timings() -> &'static Mutex<HashMap<&'static str, PerKernelStats>> {
		TIMINGS.get_or_init(|| Mutex::new(HashMap::new()))
	}

	/// Log all accumulated timing data.
	pub fn log_snapshot() {
		let timings = snapshot();
		for t in &timings {
			after_effects::log::info!(
				"[timing] {:20} {:5} avg={:7.2}ms min={:7.2}ms max={:7.2}ms last={:7.2}ms n={}",
				t.name,
				t.backend,
				t.avg_ms(),
				t.min_ms(),
				t.max_ms(),
				t.last_ms(),
				t.dispatch_count,
			);
		}
	}

	/// Record a timing measurement for a kernel dispatch.
	pub fn record(name: &'static str, backend: Backend, elapsed_ns: u64) {
		if !is_enabled() {
			return;
		}
		let mut guard = timings().lock();
		let stats = guard.entry(name).or_insert(PerKernelStats {
			backend,
			dispatch_count: 0,
			total_ns: 0,
			min_ns: u64::MAX,
			max_ns: 0,
			last_ns: 0,
		});
		stats.dispatch_count += 1;
		stats.total_ns += elapsed_ns;
		stats.min_ns = stats.min_ns.min(elapsed_ns);
		stats.max_ns = stats.max_ns.max(elapsed_ns);
		stats.last_ns = elapsed_ns;
	}

	/// Get a snapshot of all accumulated kernel timings.
	pub fn snapshot() -> Vec<KernelTiming> {
		let guard = timings().lock();
		guard
			.iter()
			.map(|(name, stats)| KernelTiming {
				name,
				backend: stats.backend,
				dispatch_count: stats.dispatch_count,
				total_ns: stats.total_ns,
				min_ns: if stats.min_ns == u64::MAX { 0 } else { stats.min_ns },
				max_ns: stats.max_ns,
				last_ns: stats.last_ns,
			})
			.collect()
	}

	/// Reset all accumulated timing data.
	pub fn reset() {
		timings().lock().clear();
	}

	/// Enable timing collection at runtime (default: enabled when feature is active).
	pub fn enable() {
		ENABLED.store(true, Ordering::Relaxed);
	}

	/// Disable timing collection at runtime.
	pub fn disable() {
		ENABLED.store(false, Ordering::Relaxed);
	}

	/// Check if timing is currently enabled.
	pub fn is_enabled() -> bool {
		ENABLED.load(Ordering::Relaxed)
	}
}

// ---------------------------------------------------------------------------
// Zero-overhead stubs when the `timing` feature is not active
// ---------------------------------------------------------------------------
#[cfg(not(feature = "timing"))]
mod imp {
	use super::{Backend, KernelTiming};

	#[inline]
	pub fn record(_name: &'static str, _backend: Backend, _elapsed_ns: u64) {}

	#[inline]
	pub fn snapshot() -> Vec<KernelTiming> {
		Vec::new()
	}

	#[inline]
	pub fn log_snapshot() {}

	#[inline]
	pub fn reset() {}

	#[inline]
	pub fn enable() {}

	#[inline]
	pub fn disable() {}

	#[inline]
	pub fn is_enabled() -> bool {
		false
	}
}

pub use imp::*;
