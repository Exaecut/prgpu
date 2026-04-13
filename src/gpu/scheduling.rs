use std::sync::atomic::{AtomicU64, Ordering};

static RENDER_GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn advance_generation() -> u64 {
    RENDER_GENERATION.fetch_add(1, Ordering::SeqCst) + 1
}

pub fn current_generation() -> u64 {
    RENDER_GENERATION.load(Ordering::SeqCst)
}

pub fn is_stale(generation: u64) -> bool {
    current_generation() > generation
}

pub fn is_latest(generation: u64) -> bool {
    current_generation() == generation
}
