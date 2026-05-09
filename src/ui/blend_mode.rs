//! Reusable blend-mode popup parameter.
//!
//! Matches the `BLEND_*` constants in `vekl/color/blend/dispatch.slang`.
//!
//! # Usage
//!
//! ```ignore
//! use prgpu::ui::{add_blend_mode_param, BlendMode};
//!
//! // In `SetupParams::setup`:
//! add_blend_mode_param(params, Params::TintBlendMode, "Tint Blend Mode", BlendMode::Multiply)?;
//!
//! // In your `kernel_params!` struct:
//! kernel_params! {
//!     MyParams for crate::params::Params {
//!         ...
//!         tint_blend_mode: u32 = [popup(TintBlendMode)];
//!     }
//! }
//!
//! // In your shader (`mode` arrives 0-based, matching BLEND_* constants):
//! float3 tinted = BlendApply(params.tintBlendMode, base.rgb, tint.rgb);
//! ```
//!
//! # Popup-indexing contract
//!
//! The `popup(V)` extractor in `prgpu::kernel_params!` normalizes AE CPU
//! (documented 1-based per Adobe PF SDK) and Premiere GPU (empirically
//! 0-based) so the kernel **always** receives a 0-based selected-index
//! (`0 = first option`, `N-1 = last`). This means:
//!
//! - `BlendMode` variant values are the same number the kernel sees.
//! - Popup option index `BLEND_MODE_OPTIONS[k]` corresponds to `BlendMode` =
//!   `k` and kernel value `k`.
//! - The vekl `BLEND_*` constants are 0-based and match exactly.
//!
//! | Popup option (`BLEND_MODE_OPTIONS[k]`) | `BlendMode` variant | `u32` kernel value |
//! |----------------------------------------|---------------------|--------------------|
//! | `"Add"` (k = 0)                        | `Add`               | 0                  |
//! | `"Multiply"` (k = 1)                   | `Multiply`          | 1                  |
//! | `"Screen"` (k = 2)                     | `Screen`            | 2                  |
//! | …                                      | …                   | …                  |
//! | `"Luminosity"` (k = 13)                | `Luminosity`        | 13                 |
//!
//! There is **no `Normal` sentinel**. To implement "no blend / pass-through",
//! pair the popup with a strength slider and skip `BlendApply` when the
//! strength is zero (see `TintStrength` in vignette / retrovhs).
//!
//! **Never apply `saturating_sub(1)` or `+ 1` to the value returned by
//! `VignetteParams::from_cpu / from_gpu` / etc. — the macro already produces
//! a host-agnostic 0-based value.**

use after_effects::{self as ae, ParamFlag, Parameters};

use crate::params::SetupParams;

/// Blend-mode discriminant. The integer value equals the 0-based popup
/// selected-index delivered to the kernel, and mirrors the `BLEND_*`
/// constants in `vekl/color/blend/dispatch.slang` byte-for-byte.
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
	/// Number of modes exposed in the popup.
	pub const COUNT: usize = 14;

	/// Kernel-side value (0-based selected-index).
	pub fn as_u32(self) -> u32 {
		self as u32
	}

	/// Popup value to pass to `PopupDef::set_default` / `set_value`. AE's
	/// PF popup API is 1-based at storage time, so we add 1.
	pub fn to_popup_value(self) -> i32 {
		(self as i32) + 1
	}

	/// Decode an AE 1-based `PopupDef.value()` into a `BlendMode`.
	/// Out-of-range values clamp to `Add` (first option).
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

/// Canonical popup option list. Order matches `BlendMode` discriminants
/// (0-based); `BLEND_MODE_OPTIONS[BlendMode::Multiply as usize] == "Multiply"`.
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
/// selection.
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
