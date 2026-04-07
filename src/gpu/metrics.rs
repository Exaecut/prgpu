use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

static FRAMES_DISPATCHED: AtomicU64 = AtomicU64::new(0);
static FRAMES_SKIPPED: AtomicU64 = AtomicU64::new(0);
static FENCE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static KERNEL_GPU_NS: AtomicU64 = AtomicU64::new(0);
static QUEUE_DEPTH: AtomicI64 = AtomicI64::new(0);

#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    pub dispatched: u64,
    pub skipped: u64,
    pub fence_wait_ns: u64,
    pub kernel_gpu_ns: u64,
    pub queue_depth: i64,
}

pub fn record_dispatch() {
    FRAMES_DISPATCHED.fetch_add(1, Ordering::Relaxed);
}

pub fn record_skip() {
    FRAMES_SKIPPED.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fence_wait_ns(ns: u64) {
    FENCE_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
}

pub fn record_kernel_gpu_ns(ns: u64) {
    KERNEL_GPU_NS.fetch_add(ns, Ordering::Relaxed);
}

pub fn inc_queue_depth() -> i64 {
    QUEUE_DEPTH.fetch_add(1, Ordering::Relaxed) + 1
}

pub fn dec_queue_depth() -> i64 {
    QUEUE_DEPTH.fetch_sub(1, Ordering::Relaxed) - 1
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        dispatched: FRAMES_DISPATCHED.load(Ordering::Relaxed),
        skipped: FRAMES_SKIPPED.load(Ordering::Relaxed),
        fence_wait_ns: FENCE_WAIT_NS.load(Ordering::Relaxed),
        kernel_gpu_ns: KERNEL_GPU_NS.load(Ordering::Relaxed),
        queue_depth: QUEUE_DEPTH.load(Ordering::Relaxed),
    }
}

pub fn reset() {
    FRAMES_DISPATCHED.store(0, Ordering::Relaxed);
    FRAMES_SKIPPED.store(0, Ordering::Relaxed);
    FENCE_WAIT_NS.store(0, Ordering::Relaxed);
    KERNEL_GPU_NS.store(0, Ordering::Relaxed);
    QUEUE_DEPTH.store(0, Ordering::Relaxed);
}
