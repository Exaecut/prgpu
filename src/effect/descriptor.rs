//! Static effect metadata exposed at registration / setup time.
//!
//! Carries the values the AE / Premiere SDKs need before any frame is
//! rendered (display name, version, options-button label, supported
//! Premiere pixel formats). Frame-dependent state belongs in `FrameData`,
//! not here.

use after_effects::pf;
use after_effects::pr;

#[derive(Clone)]
pub struct EffectDescriptor {
	pub display_name: &'static str,
	pub about_text: String,
	pub version: &'static str,
	pub options_button: Option<&'static str>,
	pub premiere_pixel_formats: Vec<pr::PixelFormat>,
}

impl EffectDescriptor {
	pub fn new(display_name: &'static str) -> Self {
		Self {
			display_name,
			about_text: display_name.to_string(),
			version: "0.0.0",
			options_button: None,
			premiere_pixel_formats: vec![pr::PixelFormat::Bgra4444_32f, pr::PixelFormat::Bgra4444_8u],
		}
	}

	pub fn about(mut self, text: impl Into<String>) -> Self {
		self.about_text = text.into();
		self
	}

	pub fn version(mut self, version: &'static str) -> Self {
		self.version = version;
		self
	}

	pub fn options_button(mut self, label: &'static str) -> Self {
		self.options_button = Some(label);
		self
	}

	pub fn premiere_pixel_formats<I: IntoIterator<Item = pr::PixelFormat>>(mut self, formats: I) -> Self {
		self.premiere_pixel_formats = formats.into_iter().collect();
		self
	}
}

/// Per-side pixel inflation. Re-exported from `cross_host` for back-compat
/// — the new `Effect::expansion` returns this same type so adapters that
/// already handle the legacy `CrossHostEffect::compute_expansion` keep
/// working.
pub use crate::effect::cross_host::ExpansionExtent;

impl ExpansionExtent {
	pub const fn none() -> Self {
		Self {
			left: 0,
			top: 0,
			right: 0,
			bottom: 0,
		}
	}

	pub const fn horizontal(px: i32) -> Self {
		Self {
			left: px,
			top: 0,
			right: px,
			bottom: 0,
		}
	}

	pub const fn vertical(px: i32) -> Self {
		Self {
			left: 0,
			top: px,
			right: 0,
			bottom: px,
		}
	}

	pub const fn ltrb(left: i32, top: i32, right: i32, bottom: i32) -> Self {
		Self { left, top, right, bottom }
	}
}

/// Ensures the AE PF parameter setup ran inside the descriptor's premiere
/// pixel-format set. Used by [`crate::adobe::ae::EffectAdapter`] during
/// `Cmd_GlobalSetup`.
pub(crate) fn install_descriptor_pixel_formats(in_data: &after_effects::InData, descriptor: &EffectDescriptor) -> Result<(), after_effects::Error> {
	if !in_data.is_premiere() {
		return Ok(());
	}
	let suite = pf::suites::PixelFormat::new()?;
	suite.clear_supported_pixel_formats(in_data.effect_ref())?;
	for fmt in &descriptor.premiere_pixel_formats {
		suite.add_supported_pixel_format(in_data.effect_ref(), *fmt)?;
	}
	pf::suites::Utility::new()?.effect_wants_checked_out_frames_to_match_render_pixel_format(in_data.effect_ref())?;
	Ok(())
}
