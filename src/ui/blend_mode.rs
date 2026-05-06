//! Reusable blend-mode popup parameter.
//!
//! Matches the discriminants in `vekl/color/blend/dispatch.slang`. To use:
//!
//! ```ignore
//! use prgpu::ui::{add_blend_mode_param, BlendMode};
//!
//! // In your `SetupParams::setup`:
//! add_blend_mode_param(params, Params::TintBlendMode, "Tint Blend Mode", BlendMode::Multiply)?;
//!
//! // In your kernel_params! struct:
//! kernel_params! {
//!     MyParams for crate::params::Params {
//!         ...
//!         tint_blend_mode: u32 = [popup(TintBlendMode)];
//!     }
//! }
//!
//! // In your shader:
//! float3 tinted = BlendApply(params.tintBlendMode, base.rgb, tint.rgb);
//! ```
//!
//! The popup is 1-indexed (Premiere/AE convention) so the discriminant of each
//! blend mode is its popup option index. `BlendMode::Normal = 0` is reserved
//! for "no blend / pass-through" and is **not** exposed in the popup list — use
//! a separate strength slider if you want to fade between source and blended.

use after_effects::{self as ae, ParamFlag, Parameters};

use crate::params::SetupParams;

/// Blend-mode discriminant. Matches `BLEND_*` constants in
/// `vekl/color/blend/dispatch.slang`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
	Normal = 0,
	Add = 1,
	Multiply = 2,
	Screen = 3,
	ColorBurn = 4,
	ColorDodge = 5,
	DarkerColor = 6,
	Overlay = 7,
	Difference = 8,
	Subtract = 9,
	Divide = 10,
	Hue = 11,
	Saturation = 12,
	Color = 13,
	Luminosity = 14,
}

impl BlendMode {
	/// Number of modes exposed in the popup (excluding `Normal`).
	pub const COUNT: usize = 14;

	pub fn as_u32(self) -> u32 {
		self as u32
	}

	/// Decode an AE/Premiere popup `value()` (1-indexed) into a `BlendMode`.
	/// Out-of-range values clamp to `Normal`.
	pub fn from_popup_value(v: i32) -> Self {
		match v {
			1 => Self::Add,
			2 => Self::Multiply,
			3 => Self::Screen,
			4 => Self::ColorBurn,
			5 => Self::ColorDodge,
			6 => Self::DarkerColor,
			7 => Self::Overlay,
			8 => Self::Difference,
			9 => Self::Subtract,
			10 => Self::Divide,
			11 => Self::Hue,
			12 => Self::Saturation,
			13 => Self::Color,
			14 => Self::Luminosity,
			_ => Self::Normal,
		}
	}

	/// Inverse of `from_popup_value`. `Normal` maps to `1` (Add) since the popup
	/// has no Normal entry — callers should pick a meaningful default explicitly.
	pub fn to_popup_value(self) -> i32 {
		match self {
			Self::Normal => 1,
			other => other as i32,
		}
	}
}

/// Canonical popup option list. The order **must** match `BlendMode`
/// discriminants 1..=14 because Premiere/AE return the 1-indexed popup
/// position as the parameter value, and slang reads it as the mode discriminant.
pub const BLEND_MODE_OPTIONS: &[&str] = &[
	"Add",
	"Multiply",
	"Screen",
	"Color Burn",
	"Color Dodge",
	"Darker Color",
	"Overlay",
	"Difference",
	"Subtract",
	"Divide",
	"Hue",
	"Saturation",
	"Color",
	"Luminosity",
];

/// Add a blend-mode popup to a `Parameters<P>`. `default` controls the initial
/// selection; `Normal` is treated as `Multiply` since the popup has no Normal.
pub fn add_blend_mode_param<P: SetupParams>(
	params: &mut Parameters<'_, P>,
	id: P,
	label: &str,
	default: BlendMode,
) -> Result<(), ae::Error> {
	let default_idx = match default {
		BlendMode::Normal => BlendMode::Multiply.to_popup_value(),
		other => other.to_popup_value(),
	};

	params.add_customized(
		id,
		label,
		ae::PopupDef::setup(|f| {
			f.set_default(default_idx);
			f.set_value(default_idx);
			f.set_options(BLEND_MODE_OPTIONS);
		}),
		|p| {
			p.set_flag(ParamFlag::SUPERVISE, true);
			-1
		},
	)
}
