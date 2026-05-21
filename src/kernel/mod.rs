//! Typed kernel descriptors.
//!
//! `declare_kernel!` emits a per-kernel module containing a `kernel()`
//! constructor that returns [`Kernel<P>`]. The graph executor calls
//! `dispatch_gpu` / `dispatch_cpu` on the descriptor based on the active
//! backend, so effect authors no longer hand-route between
//! `name(cfg, params)` and `name_cpu(...)` at every pass.

mod descriptor;
pub use descriptor::{CpuRenderFn, GpuDispatchFn, Kernel};
