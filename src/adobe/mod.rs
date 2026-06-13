//! AE / Premiere adapters. Effect crates don't touch these directly;
//! [`register_effect!`](crate::register_effect) generates the wiring.

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
