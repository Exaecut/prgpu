//! Typed kernel descriptors + GPU-ABI marker trait + macros + built-ins.
//!
//! `prgpu::declare_kernel!(name, P)` emits a per-kernel module containing a
//! `kernel()` constructor that returns [`Kernel<P>`]. The graph executor calls
//! `dispatch_gpu` / `dispatch_cpu` on the descriptor based on the active
//! backend, so effect authors no longer hand-route between
//! `name(cfg, params)` and `name_cpu(...)` at every pass.

mod descriptor;
pub mod params;
pub use descriptor::Kernel;
pub use params::KernelParams;

pub mod builtin;

mod macros;

mod from_ctx;
pub use from_ctx::FromCtx;
