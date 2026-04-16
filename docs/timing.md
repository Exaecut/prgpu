# prgpu Timing API — Design Document

## 1. Overview

A lightweight, feature-gated timing system for prgpu that provides per-kernel, per-dispatch execution timing across all backends (CPU, CUDA, Metal). Designed to validate performance improvements and diagnose regressions at runtime.

**Key principle**: Zero overhead when disabled. All timing code compiles away entirely behind `#[cfg(feature = "timing")]`.

---

## 2. Feature Flag

**File**: `prgpu/Cargo.toml`

```toml
[features]
timing = []  # Enable per-kernel dispatch timing
```

Effects opt in via their own Cargo.toml:
```toml
[dependencies]
prgpu = { path = "../prgpu", features = ["timing"] }
```

When `timing` is **not** enabled, all public API functions are inline no-ops — no `Instant::now()`, no mutex, no allocations.

---

## 3. Public API

**File**: `prgpu/src/timing.rs`

### 3.1 Types

```rust
/// Which backend produced this timing measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
}

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
        if self.dispatch_count == 0 { 0 } else { self.total_ns / self.dispatch_count }
    }
    /// Average time per dispatch in milliseconds.
    pub fn avg_ms(&self) -> f64 { self.avg_ns() as f64 / 1_000_000.0 }
    /// Minimum dispatch time in milliseconds.
    pub fn min_ms(&self) -> f64 { self.min_ns as f64 / 1_000_000.0 }
    /// Maximum dispatch time in milliseconds.
    pub fn max_ms(&self) -> f64 { self.max_ns as f64 / 1_000_000.0 }
    /// Last dispatch time in milliseconds.
    pub fn last_ms(&self) -> f64 { self.last_ns as f64 / 1_000_000.0 }
}
```

### 3.2 Functions

```rust
/// Record a timing measurement for a kernel dispatch.
/// Called from dispatch sites — not intended for user code.
pub fn record(name: &'static str, backend: Backend, elapsed_ns: u64);

/// Get a snapshot of all accumulated kernel timings.
pub fn snapshot() -> Vec<KernelTiming>;

/// Reset all accumulated timing data.
pub fn reset();

/// Enable timing collection at runtime (default: enabled when feature is active).
pub fn enable();

/// Disable timing collection at runtime.
pub fn disable();

/// Check if timing is currently enabled.
pub fn is_enabled() -> bool;
```

### 3.3 Feature-gated stubs (when `timing` is disabled)

```rust
#[cfg(not(feature = "timing"))]
pub inline fn record(_name: &'static str, _backend: Backend, _elapsed_ns: u64) {}
#[cfg(not(feature = "timing"))]
pub inline fn snapshot() -> Vec<KernelTiming> { Vec::new() }
#[cfg(not(feature = "timing"))]
pub inline fn reset() {}
#[cfg(not(feature = "timing"))]
pub inline fn enable() {}
#[cfg(not(feature = "timing"))]
pub inline fn disable() {}
#[cfg(not(feature = "timing"))]
pub inline fn is_enabled() -> bool { false }
```

---

## 4. Internal Storage

```rust
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

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
```

`record()` implementation:
```rust
pub fn record(name: &'static str, backend: Backend, elapsed_ns: u64) {
    if !is_enabled() { return; }
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
```

---

## 5. CPU Timing Insertion

### 5.1 `render_cpu()` signature change

**File**: `prgpu/src/cpu/render.rs`

```diff
-pub fn render_cpu<P: Copy + Sync>(
+pub fn render_cpu<P: Copy + Sync>(
+    kernel_name: &'static str,
     in_data: &ae::InData,
     in_layer: &ae::Layer,
     out_layer: &mut ae::Layer,
     config: &Configuration,
     dispatch_fn: CpuDispatchFn,
     user_params: &P,
 ) -> Result<(), ae::Error> {
```

### 5.2 Timing around dispatch

```rust
pub fn render_cpu<P: Copy + Sync>(
    kernel_name: &'static str,
    in_data: &ae::InData,
    in_layer: &ae::Layer,
    out_layer: &mut ae::Layer,
    config: &Configuration,
    dispatch_fn: CpuDispatchFn,
    user_params: &P,
) -> Result<(), ae::Error> {
    let w = config.width;
    let h = config.height;
    if w == 0 || h == 0 {
        return Ok(());
    }

    // ... existing buffer setup ...

    let start = std::time::Instant::now();  // only compiled with feature

    let result = if can_iterate_with {
        ae_dispatch(in_layer, out_layer, buffers, tp, user_params, dispatch_fn)
    } else {
        // ... existing out_buf setup ...
        rayon_dispatch(w, h, buffers, tp, user_params, dispatch_fn, out_buf, in_buf, out_stride_bytes)
    };

    crate::timing::record(kernel_name, crate::timing::Backend::Cpu, start.elapsed().as_nanos() as u64);

    result
}
```

### 5.3 `declare_kernel!` macro change

**File**: `prgpu/src/kernels/mod.rs`

The CPU dispatch section changes to pass the kernel name:

```rust
$crate::cpu::render::render_cpu(
    stringify!($name),   // ← NEW: kernel name for timing
    in_data,
    in_layer,
    out_layer,
    config,
    dispatch_fn,
    &user_params,
)
```

---

## 6. GPU Timing Insertion

### 6.1 CUDA — Event-based timing

**File**: `prgpu/src/gpu/backends/cuda/mod.rs`

CUDA provides stream-accurate GPU timing via `cuEventRecord` / `cuEventElapsedTime`.

**Event cache** (reuse events per context to avoid allocation overhead):

```rust
use std::sync::OnceLock;
use parking_lot::Mutex;
use cudarc::driver::sys as cu;

static EVENT_CACHE: OnceLock<Mutex<HashMap<usize, (cu::CUevent, cu::CUevent)>>> = OnceLock::new();

fn event_cache() -> &'static Mutex<HashMap<usize, (cu::CUevent, cu::CUevent)>> {
    EVENT_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_create_events(ctx: usize) -> (cu::CUevent, cu::CUevent) {
    let mut guard = event_cache().lock();
    *guard.entry(ctx).or_insert_with(|| {
        let mut start: cu::CUevent = std::ptr::null_mut();
        let mut stop: cu::CUevent = std::ptr::null_mut();
        unsafe {
            cu::cuEventCreate(&mut start, cu::CUevent_flags::CU_EVENT_DEFAULT);
            cu::cuEventCreate(&mut stop, cu::CUevent_flags::CU_EVENT_DEFAULT);
        }
        (start, stop)
    })
}
```

**Timing in `run()`**:

```rust
pub fn run<UP>(config: &Configuration, user_params: UP, ...) -> Result<(), &'static str> {
    // ... existing setup ...

    let (start_event, stop_event) = get_or_create_events(ctx as usize);

    unsafe {
        cu::cuEventRecord(start_event, stream);
        dispatch(ctx, stream, func, grid_x, grid_y, block_x, block_y, &mut params)?;
        cu::cuEventRecord(stop_event, stream);
    }

    // ... existing spin-loop wait for completion ...

    // After stream completion, read GPU timing
    let mut gpu_ms: f32 = 0.0;
    unsafe {
        cu::cuEventElapsedTime(&mut gpu_ms, start_event, stop_event);
    }
    crate::timing::record(entry, crate::timing::Backend::Cuda, (gpu_ms * 1_000_000.0) as u64);

    Ok(())
}
```

**Cleanup**: Destroy cached events in `cuda::pipeline::cleanup()`:
```rust
for (_, (start, stop)) in event_cache().lock().drain() {
    unsafe {
        cu::cuEventDestroy_v2(start);
        cu::cuEventDestroy_v2(stop);
    }
}
```

### 6.2 Metal — Command buffer timing

**File**: `prgpu/src/gpu/backends/metal/mod.rs`

Metal already computes `gpu_ms` from `GPUStartTime`/`GPUEndTime`. Just add the recording call:

```rust
// After existing GPU time computation:
let gpu_start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
let gpu_end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
let gpu_ms = (gpu_end - gpu_start) * 1000.0;

crate::timing::record(entry, crate::timing::Backend::Metal, (gpu_ms * 1_000_000.0) as u64);
```

---

## 7. Module Registration

**File**: `prgpu/src/lib.rs`

```rust
#[cfg(feature = "timing")]
pub mod timing;

#[cfg(not(feature = "timing"))]
pub mod timing {
    // Stubs that compile away
    use crate::timing_types;  // or inline the types
    pub fn record(_name: &'static str, _backend: super::timing::Backend, _elapsed_ns: u64) {}
    pub fn snapshot() -> Vec<super::timing::KernelTiming> { Vec::new() }
    pub fn reset() {}
    pub fn enable() {}
    pub fn disable() {}
    pub fn is_enabled() -> bool { false }
}
```

Actually, simpler approach — use `cfg` inside the single `timing.rs` file:

```rust
// prgpu/src/timing.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend { Cpu, Cuda, Metal }

#[derive(Debug, Clone)]
pub struct KernelTiming { ... }

#[cfg(feature = "timing")]
mod imp {
    // Full implementation with storage, recording, etc.
}

#[cfg(not(feature = "timing"))]
mod imp {
    // No-op stubs
}

pub use imp::*;
```

---

## 8. Developer Experience

### Usage in vignette (or any effect)

```rust
// In GlobalSetup or FrameSetup:
#[cfg(feature = "timing")]
prgpu::timing::enable();

// After rendering a frame (e.g., in FrameSetdown or a debug UI):
#[cfg(feature = "timing")]
{
    let timings = prgpu::timing::snapshot();
    for t in &timings {
        log::info!(
            "[timing] {:20s} {:5s} avg={:7.2}ms min={:7.2}ms max={:7.2}ms n={}",
            t.name,
            match t.backend {
                prgpu::timing::Backend::Cpu => "CPU",
                prgpu::timing::Backend::Cuda => "CUDA",
                prgpu::timing::Backend::Metal => "Metal",
            },
            t.avg_ms(), t.min_ms(), t.max_ms(), t.dispatch_count,
        );
    }
}

// Before benchmark comparison:
prgpu::timing::reset();
```

### Example output

```
[timing] blur                 CPU   avg=  12.34ms min=  11.98ms max=  14.02ms n=3
[timing] vignette             CPU   avg=   3.45ms min=   3.21ms max=   4.10ms n=1
[timing] blur                 CUDA  avg=   0.87ms min=   0.82ms max=   1.05ms n=3
[timing] vignette             CUDA  avg=   0.12ms min=   0.11ms max=   0.15ms n=1
```

---

## 9. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `prgpu/Cargo.toml` | Modify | Add `timing = []` feature |
| `prgpu/src/timing.rs` | **Create** | Timing module with types, storage, API |
| `prgpu/src/lib.rs` | Modify | Add `pub mod timing;` |
| `prgpu/src/cpu/render.rs` | Modify | Add `kernel_name` param + `Instant` timing |
| `prgpu/src/kernels/mod.rs` | Modify | Pass `stringify!($name)` in CPU dispatch |
| `prgpu/src/gpu/backends/cuda/mod.rs` | Modify | Add CUDA event timing in `run()` |
| `prgpu/src/gpu/backends/cuda/pipeline.rs` | Modify | Destroy cached events in `cleanup()` |
| `prgpu/src/gpu/backends/metal/mod.rs` | Modify | Add `timing::record()` in `run()` |
| `prgpu/src/gpu/metrics.rs` | No change | Existing aggregate metrics remain separate |

---

## 10. Implementation Order

1. **`prgpu/Cargo.toml`** — add feature flag
2. **`prgpu/src/timing.rs`** — create module with types + stubs + impl
3. **`prgpu/src/lib.rs`** — register module
4. **`prgpu/src/cpu/render.rs`** — add `kernel_name` + `Instant` timing
5. **`prgpu/src/kernels/mod.rs`** — pass kernel name in macro
6. **`prgpu/src/gpu/backends/metal/mod.rs`** — add Metal timing record
7. **`prgpu/src/gpu/backends/cuda/mod.rs`** — add CUDA event timing
8. **`prgpu/src/gpu/backends/cuda/pipeline.rs`** — event cleanup
9. **Build & test** — verify `--features timing` compiles, and default build still works

---

## 11. Benchmark Validation Plan

Once the timing API is implemented, use it to validate the previous optimizations:

```rust
// Before optimization (revert changes temporarily):
prgpu::timing::reset();
// Render 20 frames...
let before = prgpu::timing::snapshot();

// After optimization:
prgpu::timing::reset();
// Render 20 frames...
let after = prgpu::timing::snapshot();

// Compare:
for (b, a) in before.iter().zip(after.iter()) {
    let speedup = b.avg_ns() as f64 / a.avg_ns() as f64;
    log::info!("[bench] {:20s} {:.2}x speedup", b.name, speedup);
}
```
