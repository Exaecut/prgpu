//! Adobe SDK adapters that bridge the [`crate::effect::Effect`] trait into
//! the host-specific traits the existing prgpu macros (`ae::define_effect!`
//! and `pr::define_gpu_filter!`) expect.
//!
//! Effect crates declare:
//!
//! ```ignore
//! pub type Plugin = prgpu::adobe::ae::EffectAdapter<MyEffect>;
//! ae::define_effect!(Plugin, (), MyParams);
//!
//! pub type PremiereGPU = prgpu::adobe::premiere::GpuFilterAdapter<MyEffect>;
//! pr::define_gpu_filter!(PremiereGPU);
//! ```
//!
//! The adapters own the AE PF / Premiere GPU lifecycle (license init,
//! parameter visibility flips, expansion calculation, source snapshot,
//! graph execution) so per-effect code stays focused on the algorithm.

pub mod ae;
pub mod premiere;
