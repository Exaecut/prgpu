//! Shared helpers for radial / angular sweep effects (radial blur, motion blur, lens streaks, ...).
//!
//! The kernel loop bound must be uniform across the dispatch, so the host picks
//! a single sample count from the user's quality knob and the geometric trail length.

/// Lower bound on samples. Two = anchor + far-end blend, the "do nothing" floor at quality = 0%.
pub const MIN_SAMPLES: u32 = 2;

/// Upper bound on samples. 1024 × 4K already costs ~8 G fetches per frame, well past where visual benefit plateaus.
pub const MAX_SAMPLES: u32 = 1024;

#[derive(Debug, Clone, Copy)]
pub struct SweepSamplesInputs {
	pub width: u32,
	pub height: u32,
	/// Worst-case screen-space trail length, in normalized UV (0..1 across the longest axis).
	/// Angular sweep: `2 * sin(angle/2)`. Radial (zoom): `|distance|`.
	pub trail_uv: f32,
	/// Quality knob, percent. 100% = Nyquist (1 sample per pixel of trail), 200% = 2× over-Nyquist.
	pub quality_pct: f32,
}

/// Sample count for an angular / radial sweep so spacing stays under one pixel
/// at the worst-case trail at quality = 100% (Nyquist for non-aliased sweep).
///
/// Linear in quality so 50% halves and 200% doubles, matching artist expectations.
pub fn compute_sweep_samples(inputs: SweepSamplesInputs) -> u32 {
	let max_axis = inputs.width.max(inputs.height) as f32;
	let trail_px = (inputs.trail_uv.abs() * max_axis).max(0.0);
	let q = (inputs.quality_pct / 100.0).max(0.0);
	let target = (q * trail_px).round() as i64 + MIN_SAMPLES as i64;
	target.clamp(MIN_SAMPLES as i64, MAX_SAMPLES as i64) as u32
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn floor_at_zero_quality() {
		let n = compute_sweep_samples(SweepSamplesInputs {
			width: 1920,
			height: 1080,
			trail_uv: 0.5,
			quality_pct: 0.0,
		});
		assert_eq!(n, MIN_SAMPLES);
	}

	#[test]
	fn nyquist_at_full_quality() {
		let n = compute_sweep_samples(SweepSamplesInputs {
			width: 1920,
			height: 1080,
			trail_uv: 1.0,
			quality_pct: 100.0,
		});
		assert_eq!(n, 1024); // clamped to MAX_SAMPLES
	}

	#[test]
	fn moderate_trail_scales_linearly() {
		let n100 = compute_sweep_samples(SweepSamplesInputs {
			width: 1920,
			height: 1080,
			trail_uv: 0.05,
			quality_pct: 100.0,
		});
		let n200 = compute_sweep_samples(SweepSamplesInputs {
			width: 1920,
			height: 1080,
			trail_uv: 0.05,
			quality_pct: 200.0,
		});
		assert_eq!(n100, 96 + 2);
		assert_eq!(n200, 192 + 2);
	}

	#[test]
	fn never_below_min() {
		let n = compute_sweep_samples(SweepSamplesInputs {
			width: 4,
			height: 4,
			trail_uv: 0.0,
			quality_pct: 200.0,
		});
		assert!(n >= MIN_SAMPLES);
	}

	#[test]
	fn clamps_to_max() {
		let n = compute_sweep_samples(SweepSamplesInputs {
			width: 4096,
			height: 2160,
			trail_uv: 1.0,
			quality_pct: 200.0,
		});
		assert_eq!(n, MAX_SAMPLES);
	}
}
