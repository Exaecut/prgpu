//! Bounded rayon global pool for CPU render dispatches.
//!
//! Rationale
//! ---------
//! Premiere renders many frames in parallel (observed `concurrent` up to 16).
//! The default rayon global pool has `num_cpus` workers. That is fine for a
//! single dispatch but becomes problematic when Premiere itself is CPU-bound
//! on UI, audio, and layout threads — they get starved of cores.
//!
//! We re-initialize the rayon **global** pool once with a bounded worker
//! count (default: `max(1, num_cpus - 2)`). `par_chunks_mut(...).for_each(...)`
//! then naturally runs on that bounded pool without needing `install()`
//! wrappers (which, with N > workers concurrent installs, serialize the
//! outer closures and starve the inner fork-join — exactly the 20x regression
//! observed when we tried `install()`).
//!
//! Overridable at runtime via `EX_RENDER_WORKERS`.

use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

static INIT: Once = Once::new();
static ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Default policy: leave two logical cores for the host UI/audio/layout
/// threads. `rayon::current_num_threads()` is safe to call before the global
/// pool is installed.
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

/// Initialize the rayon global pool with a bounded worker count. Safe to
/// invoke multiple times — the underlying `build_global()` is guarded by
/// `Once`. If another library already installed the global pool, we fall
/// back silently and still report the observed worker count.
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

/// Number of workers active in the render pool. Reported in the per-dispatch
/// diagnostic line so we can verify the policy is active.
#[inline]
pub fn worker_count() -> usize {
	ensure_initialized();
	ACTIVE_WORKERS.load(Ordering::Relaxed).max(1)
}
