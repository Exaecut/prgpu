use std::sync::atomic::{AtomicU64, Ordering};

static RENDER_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Advances the global render generation and returns the new value.
/// Each call to `Configuration::effect()` / `transition()` calls this once.
pub fn advance_generation() -> u64 {
    RENDER_GENERATION.fetch_add(1, Ordering::SeqCst) + 1
}

pub fn current_generation() -> u64 {
    RENDER_GENERATION.load(Ordering::SeqCst)
}

/// A frame is stale if a strictly newer render request exists.
pub fn is_stale(generation: u64) -> bool {
    current_generation() > generation
}

/// A frame is the latest if no newer request has been made.
pub fn is_latest(generation: u64) -> bool {
    current_generation() == generation
}
