//! Blend-mode popup options.
//!
//! Discriminants mirror the `BLEND_*` constants in
//! `vekl/color/blend/dispatch.slang` byte-for-byte: for any variant,
//! `BlendMode as u32`, `LABELS[k]`, the slang `BLEND_*` constant, and the
//! kernel `u32` are byte-equal. Sugar attribute `#[blend_mode(...)]` in
//! `params!` expands to `popup(options = prgpu::BlendMode, ...)`.

use crate::Popup;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Popup)]
pub enum BlendMode {
	#[option("Add")]
	Add = 0,
	#[option("Multiply")]
	Multiply = 1,
	#[option("Screen")]
	Screen = 2,
	#[option("Color Burn")]
	ColorBurn = 3,
	#[option("Color Dodge")]
	ColorDodge = 4,
	#[option("Darker Color")]
	DarkerColor = 5,
	#[option("Overlay")]
	Overlay = 6,
	#[option("Difference")]
	Difference = 7,
	#[option("Subtract")]
	Subtract = 8,
	#[option("Divide")]
	Divide = 9,
	#[option("Hue")]
	Hue = 10,
	#[option("Saturation")]
	Saturation = 11,
	#[option("Color")]
	Color = 12,
	#[option("Luminosity")]
	Luminosity = 13,
}
