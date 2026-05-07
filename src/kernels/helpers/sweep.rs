//! Shared helpers for radial / angular sweep effects (radial blur, motion
//! blur, lens streaks, etc.).
//!
//! The host has to pick a single sample count for the whole dispatch (it
//! drives the kernel loop bound, which must be a uniform). Picking it from
//! the user's quality knob and the geometric trail length keeps the visual
//! result band-free without forcing the user to fiddle with a "samples"
//! slider.

/// Smallest sample count we ever dispatch. Two samples = a simple
/// linear blend of the anchor and the far end, which is barely a blur but
/// still useful as a "do basically nothing" floor at quality = 0 %.
pub const MIN_SAMPLES: u32 = 2;

/// Clamp the upper end so a runaway slider can't tank performance. 1024
/// samples per pixel of a 4K dispatch is already ~8 G fetches per frame —
/// well past where the visual benefit plateaus.
pub const MAX_SAMPLES: u32 = 1024;

/// Inputs for [`compute_sweep_samples`].
#[derive(Debug, Clone, Copy)]
pub struct SweepSamplesInputs {
	/// Output frame width in pixels.
	pub width: u32,
	/// Output frame height in pixels.
	pub height: u32,
	/// Maximum *screen-space* trail length expected, in normalized UV units
	/// (0..1 across the longest axis). For an angular sweep this is the
	/// chord length at the farthest pixel: `2 * sin(angle/2)`. For a radial
	/// (zoom) sweep this is `|distance|`.
	pub trail_uv: f32,
	/// User's quality knob, in percent. 0 % = floor (`MIN_SAMPLES`),
	/// 100 % = Nyquist (1 sample per pixel of trail), 200 % = 2× over-Nyquist.
	pub quality_pct: f32,
}

/// Pick a sample count for an angular / radial sweep so that even at the
/// pixel with the longest trail, sample spacing stays under one pixel
/// at quality = 100 % (Nyquist for a non-aliased sweep).
///
/// Quality is linear, so 50 % halves the count and 200 % doubles it —
/// matches what artists expect from a percentage knob.
pub fn compute_sweep_samples(inputs: SweepSamplesInputs) -> u32 {
	let max_axis = inputs.width.max(inputs.height) as f32;
	// Trail length in pixels at the worst-case pixel.
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
		// 1920px wide, trail = 1.0 UV → 1920 px trail. At 100% we want ~1922
		// (rounded + MIN_SAMPLES).
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
		// 1920px wide, trail = 0.05 UV → 96 px. At 100% we want ~98.
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
