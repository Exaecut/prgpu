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

/// Adobe high-precision time base: 254,016,000,000 ticks per second (`PrTime`).
/// The SDK does not expose this as a constant.
pub const PR_TICKS_PER_SECOND: f64 = 254_016_000_000.0;

/// `PrTime` ticks → seconds. The canonical `frame.time` unit fed to every
/// backend; both hosts report sequence time in this base.
pub fn ticks_to_seconds(ticks: i64) -> f32 {
	(ticks as f64 / PR_TICKS_PER_SECOND) as f32
}
