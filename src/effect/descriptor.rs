//! Static effect metadata exposed at registration / setup time.
//!
//! Carries the values the AE / Premiere SDKs need before any frame is
//! rendered (display name, version, options-button label, supported
//! Premiere pixel formats). Frame-dependent state belongs in `FrameData`,
//! not here.

use after_effects::Rect;
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


/// Per-side pixel inflation applied uniformly to the input layer to compute
/// the rendered output rect.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpansionExtent {
	pub left: i32,
	pub top: i32,
	pub right: i32,
	pub bottom: i32,
}

pub enum ExpansionSymmetry {
	IndependentAxes { horizontal: i32, vertical: i32 },
	Symmetric(i32),
}

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

	/// Symmetric per-side inflation with independent horizontal and vertical
	/// extents: `horizontal` on left/right, `vertical` on top/bottom. For effects whose
	/// X and Y growth differ (e.g. separate horizontal/vertical shake budgets).
	pub const fn symetric(symetry: ExpansionSymmetry) -> Self {
		match symetry {
			ExpansionSymmetry::IndependentAxes { horizontal, vertical } => Self { left: horizontal, top: vertical, right: horizontal, bottom: vertical },
			ExpansionSymmetry::Symmetric(px) => Self { left: px, top: px, right: px, bottom: px },
		}
	}

	pub fn is_zero(&self) -> bool {
		self.left == 0 && self.top == 0 && self.right == 0 && self.bottom == 0
	}

	pub fn total_width(&self) -> i32 {
		self.left + self.right
	}

	pub fn total_height(&self) -> i32 {
		self.top + self.bottom
	}

	pub fn inflate_rect(&self, r: Rect) -> Rect {
		let mut out = Rect::empty();
		out.left = r.left - self.left;
		out.top = r.top - self.top;
		out.right = r.right + self.right;
		out.bottom = r.bottom + self.bottom;
		out
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

#[cfg(test)]
mod tests {
	use super::ExpansionExtent;

	#[test]
	fn symmetric_hv_independent_axes() {
		let e = ExpansionExtent::symetric(ExpansionSymmetry::IndependentAxes { horizontal: 30, vertical: 12 });
		assert_eq!((e.left, e.right), (30, 30));
		assert_eq!((e.top, e.bottom), (12, 12));
		assert_eq!(e.total_width(), 60);
		assert_eq!(e.total_height(), 24);
	}

	#[test]
	fn symmetric_hv_matches_symmetric_when_equal() {
		assert_eq!(
			ExpansionExtent::symetric(ExpansionSymmetry::IndependentAxes { horizontal: 20, vertical: 20 }).total_width(),
			ExpansionExtent::symetric(ExpansionSymmetry::Symmetric(20)).total_width()
		);
	}
}
