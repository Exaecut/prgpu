//! Per-kernel dispatch timing for CPU and GPU backends.
//!
//! Enable via `features = ["timing"]`; otherwise every public function is a no-op.

pub use crate::types::Backend;

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
	pub fn avg_ns(&self) -> u64 {
		if self.dispatch_count == 0 {
			0
		} else {
			self.total_ns / self.dispatch_count
		}
	}

	pub fn avg_ms(&self) -> f64 {
		self.avg_ns() as f64 / 1_000_000.0
	}

	pub fn min_ms(&self) -> f64 {
		self.min_ns as f64 / 1_000_000.0
	}

	pub fn max_ms(&self) -> f64 {
		self.max_ns as f64 / 1_000_000.0
	}

	pub fn last_ms(&self) -> f64 {
		self.last_ns as f64 / 1_000_000.0
	}
}

#[cfg(feature = "timing")]
mod imp {
	use super::{Backend, KernelTiming};
	use parking_lot::Mutex;
	use std::collections::HashMap;
	use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
	use std::sync::OnceLock;

	static ENABLED: AtomicBool = AtomicBool::new(true);

	/// Throttle for `log_snapshot()`. With `60` we emit ~once per second at 60 fps,
	/// dropping `OutputDebugStringW` / `DBWinMutex` contention that otherwise dominates
	/// wall-clock variance in Premiere. `0` disables throttling. Default: 60.
	static LOG_SNAPSHOT_INTERVAL: AtomicU64 = AtomicU64::new(60);
	static LOG_SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

	pub fn set_log_snapshot_interval(interval: u64) {
		LOG_SNAPSHOT_INTERVAL.store(interval, Ordering::Relaxed);
	}

	/// Emit an aggregated snapshot now, ignoring the throttle counter.
	pub fn log_snapshot_now() {
		emit_snapshot();
	}

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

	/// Emit accumulated timings, throttled by `set_log_snapshot_interval`. Use `log_snapshot_now` for an unconditional emit.
	pub fn log_snapshot() {
		let interval = LOG_SNAPSHOT_INTERVAL.load(Ordering::Relaxed);
		if interval == 0 {
			emit_snapshot();
			return;
		}
		let prev = LOG_SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed);
		if prev % interval == 0 {
			emit_snapshot();
		}
	}

	#[inline]
	fn emit_snapshot() {
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

	pub fn reset() {
		timings().lock().clear();
	}

	/// Enable timing collection (default: enabled when feature is active).
	pub fn enable() {
		ENABLED.store(true, Ordering::Relaxed);
	}

	pub fn disable() {
		ENABLED.store(false, Ordering::Relaxed);
	}

	pub fn is_enabled() -> bool {
		ENABLED.load(Ordering::Relaxed)
	}
}

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
	pub fn log_snapshot_now() {}

	#[inline]
	pub fn set_log_snapshot_interval(_interval: u64) {}

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
