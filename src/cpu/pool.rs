//! Bounded global rayon pool for CPU render dispatches.
//!
//! Premiere renders many frames in parallel (observed `concurrent` up to 16) and
//! the default rayon pool starves the host's UI / audio / layout threads. We
//! re-init the rayon **global** pool once with `max(1, num_cpus - 2)` workers so
//! `par_chunks_mut(...).for_each(...)` runs on the bounded pool without needing
//! `install()` (which serializes outer closures and was a 20× regression in tests).
//!
//! Override at runtime via `EX_RENDER_WORKERS`.

use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

static INIT: Once = Once::new();
static ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Default: leave two cores for the host's UI/audio/layout threads.
fn default_worker_count() -> usize {
	rayon::current_num_threads().saturating_sub(2).max(1)
}

fn configured_worker_count() -> usize {
	if let Ok(v) = std::env::var("EX_RENDER_WORKERS")
		&& let Ok(n) = v.parse::<usize>()
		&& n > 0
	{
		return n;
	}
	default_worker_count()
}

/// Initialize the rayon global pool once. Safe to call repeatedly; if another lib already installed the global pool, we silently fall back.
pub fn ensure_initialized() {
	INIT.call_once(|| {
		let workers = configured_worker_count();
		let result = rayon::ThreadPoolBuilder::new()
			.num_threads(workers)
			.thread_name(|i| format!("ex-render-{i}"))
			.build_global();
		let active = if result.is_ok() {
			workers
		} else {
			rayon::current_num_threads()
		};
		ACTIVE_WORKERS.store(active, Ordering::Relaxed);
	});
}

/// Active worker count, reported in the per-dispatch diagnostic so the policy can be verified at runtime.
#[inline]
pub fn worker_count() -> usize {
	ensure_initialized();
	ACTIVE_WORKERS.load(Ordering::Relaxed).max(1)
}
