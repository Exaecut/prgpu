//! Reusable blend-mode popup parameter.
//!
//! Mirrors the `BLEND_*` constants in `vekl/color/blend/dispatch.slang` byte-for-byte.
//!
//! ```ignore
//! use prgpu::ui::{add_blend_mode_param, BlendMode};
//!
//! // setup:
//! add_blend_mode_param(params, Params::TintBlendMode, "Tint Blend Mode", BlendMode::Multiply)?;
//!
//! // kernel_params:
//! tint_blend_mode: u32 = [popup(TintBlendMode)];
//!
//! // shader:
//! float3 tinted = BlendApply(params.tintBlendMode, base.rgb, tint.rgb);
//! ```
//!
//! For any variant, `BlendMode as u32`, `BLEND_MODE_OPTIONS[k]`, the slang
//! `BLEND_*` constant, and the kernel `u32` are byte-equal. The `popup(V)`
//! extractor handles the AE-vs-Premiere host conversion internally.
//!
//! "No blend / pass-through" is a separate strength slider, not a mode: skip
//! `BlendApply` at strength = 0 or `lerp(base, BlendApply(...), strength)` to fade.

use after_effects::{self as ae, ParamFlag, Parameters};

use crate::params::SetupParams;

/// Blend-mode discriminant. The integer value equals the 0-based popup index
/// delivered to the kernel and the `BLEND_*` constants in `vekl/color/blend/dispatch.slang`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
	Add = 0,
	Multiply = 1,
	Screen = 2,
	ColorBurn = 3,
	ColorDodge = 4,
	DarkerColor = 5,
	Overlay = 6,
	Difference = 7,
	Subtract = 8,
	Divide = 9,
	Hue = 10,
	Saturation = 11,
	Color = 12,
	Luminosity = 13,
}

impl BlendMode {
	pub const COUNT: usize = 14;

	pub fn as_u32(self) -> u32 {
		self as u32
	}

	/// Popup value for `PopupDef::set_default` / `set_value`. AE PF popups are 1-based at storage, so we add 1.
	pub fn to_popup_value(self) -> i32 {
		(self as i32) + 1
	}

	/// Decode an AE 1-based `PopupDef.value()`. Out-of-range clamps to `Add`.
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
			_ => Self::Add,
		}
	}
}

/// Popup option list. Order matches `BlendMode` discriminants 0-based; `BLEND_MODE_OPTIONS[BlendMode::Multiply as usize] == "Multiply"`.
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

/// Add a blend-mode popup to `Parameters<P>` with `default` selected.
pub fn add_blend_mode_param<P: SetupParams>(
	params: &mut Parameters<'_, P>,
	id: P,
	label: &str,
	default: BlendMode,
) -> Result<(), ae::Error> {
	let default_popup = default.to_popup_value();

	params.add_customized(
		id,
		label,
		ae::PopupDef::setup(|f| {
			f.set_default(default_popup);
			f.set_value(default_popup);
			f.set_options(BLEND_MODE_OPTIONS);
		}),
		|p| {
			p.set_flag(ParamFlag::SUPERVISE, true);
			-1
		},
	)
}
