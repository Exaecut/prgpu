//! Shared blur-pipeline helpers for separable Gaussian effects.
//!
//! Provides a single, self-explanatory entry point for picking the
//! intermediate buffer size when ping-ponging horizontal/vertical
//! Gaussian passes through a downsampled buffer.

/// Floor of the radius-driven scale factor.
///
/// We never let the radius factor pull the buffer below 25 % of full size on
/// each axis (1/16 of the area) — past that, even strong Gaussians start to
/// reveal aliasing in motion (a.k.a. "swimming" / chunky bokeh) on smooth
/// gradients, which the user *would* notice.
const RADIUS_FACTOR_FLOOR: f32 = 0.25;

/// Target effective sigma in the *downsampled* space, in texels.
///
/// A Gaussian with σ ≥ ~2 downsampled-texels keeps high-frequency energy
/// well below the post-downsample Nyquist limit, so downsampling further
/// would just clip frequencies the blur was already removing — visually
/// imperceptible. Below 2 the kernel becomes narrow enough that the
/// downsample artifacts (blocky bokeh, swimming on motion) start to show.
const TARGET_DOWNSAMPLED_SIGMA: f32 = 2.0;

/// Inputs for the downsample size computation.
#[derive(Debug, Clone, Copy)]
pub struct BlurDownsampleInputs {
	/// Full-resolution buffer width, in texels.
	pub width: u32,
	/// Full-resolution buffer height, in texels.
	pub height: u32,
	/// Effective Gaussian sigma in *full-resolution texels* (post host-downsample).
	/// In the typical Premiere/AE flow this is `blur_radius / host_downsample_x`.
	pub sigma_full: f32,
	/// User-controlled "Quality" slider in `[0, 100]` (percent of full resolution).
	pub quality_pct: f32,
}

/// Output of the downsample size computation.
#[derive(Debug, Clone, Copy)]
pub struct BlurDownsample {
	/// Downsampled width, ≥ 1 texel.
	pub width: u32,
	/// Downsampled height, ≥ 1 texel.
	pub height: u32,
	/// Quality factor that was applied (clamped `[0.01, 1.0]`).
	pub quality_factor: f32,
	/// Radius-driven factor that was applied (clamped `[FLOOR, 1.0]`).
	pub radius_factor: f32,
}

impl BlurDownsample {
	/// Combined scale that was applied to each axis (`quality_factor * radius_factor`).
	pub fn combined_factor(&self) -> f32 {
		self.quality_factor * self.radius_factor
	}
}

/// Map a full-resolution sigma to a `[FLOOR, 1.0]` scale factor that hides
/// resolution loss inside the blur's own low-pass response.
///
/// The kernel removes frequencies above `~1/(2πσ)`. After downsampling by
/// `s` the new Nyquist is `0.5/s`, and we keep ~2× headroom so the rolled-off
/// tail doesn't alias — equivalently, σ in downsampled space stays ≥
/// `TARGET_DOWNSAMPLED_SIGMA`. Solving `σ_full × s ≥ target` gives
/// `s ≥ target / σ_full`, which is exactly the formula below before clamping.
#[inline]
pub fn radius_scale_factor(sigma_full: f32) -> f32 {
	if sigma_full <= TARGET_DOWNSAMPLED_SIGMA {
		// Tiny blur — any extra downsample would show stair-stepping before
		// the kernel could mask it.
		return 1.0;
	}
	(TARGET_DOWNSAMPLED_SIGMA / sigma_full).clamp(RADIUS_FACTOR_FLOOR, 1.0)
}

/// Compute the intermediate buffer size for a separable Gaussian blur given
/// the user's quality preference and the actual blur strength.
///
/// `final_dims = full_dims × quality_factor × radius_factor`, where:
/// - `quality_factor = clamp(quality_pct / 100, 0.01, 1.0)` — explicit user knob.
/// - `radius_factor = radius_scale_factor(sigma_full)` — automatic, see fn docs.
///
/// Both factors are reported back so callers can log / display them.
pub fn compute_downsample(inputs: BlurDownsampleInputs) -> BlurDownsample {
	let quality_factor = (inputs.quality_pct / 100.0).clamp(0.01, 1.0);
	let radius_factor = radius_scale_factor(inputs.sigma_full);
	let scale = quality_factor * radius_factor;

	let w = ((inputs.width as f32 * scale) as u32).max(1);
	let h = ((inputs.height as f32 * scale) as u32).max(1);

	BlurDownsample {
		width: w,
		height: h,
		quality_factor,
		radius_factor,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn approx_eq(a: f32, b: f32) -> bool {
		(a - b).abs() < 1e-5
	}

	#[test]
	fn tiny_sigma_keeps_full_res() {
		assert!(approx_eq(radius_scale_factor(0.5), 1.0));
		assert!(approx_eq(radius_scale_factor(2.0), 1.0));
	}

	#[test]
	fn medium_sigma_halves_resolution() {
		// σ = 4 → factor = 2/4 = 0.5
		assert!(approx_eq(radius_scale_factor(4.0), 0.5));
	}

	#[test]
	fn huge_sigma_clamps_to_floor() {
		assert!(approx_eq(radius_scale_factor(64.0), RADIUS_FACTOR_FLOOR));
		assert!(approx_eq(radius_scale_factor(1024.0), RADIUS_FACTOR_FLOOR));
	}

	#[test]
	fn combined_factors_multiply() {
		let ds = compute_downsample(BlurDownsampleInputs {
			width: 1920,
			height: 1080,
			sigma_full: 8.0,
			quality_pct: 50.0,
		});
		// quality = 0.5, radius = 2/8 = 0.25, combined = 0.125
		assert!(approx_eq(ds.quality_factor, 0.5));
		assert!(approx_eq(ds.radius_factor, 0.25));
		assert!(approx_eq(ds.combined_factor(), 0.125));
		assert_eq!(ds.width, 240);
		assert_eq!(ds.height, 135);
	}

	#[test]
	fn min_one_texel() {
		let ds = compute_downsample(BlurDownsampleInputs {
			width: 4,
			height: 4,
			sigma_full: 256.0,
			quality_pct: 1.0,
		});
		assert!(ds.width >= 1 && ds.height >= 1);
	}
}
