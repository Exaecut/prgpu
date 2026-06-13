//! Normalized per-parameter values captured at snapshot time.
//!
//! Every host quirk (8-bit colour channels, 1-based AE popups, pixel-space
//! points) is resolved into one of these variants once, so the read side
//! (`Ctx::get`) is host-agnostic.

use crate::types::Pixel;

/// RGBA in 0–1 linear-host channel order (R=red, A=alpha), already normalized
/// from the host's 8-bit `PF_Pixel`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
	pub r: f32,
	pub g: f32,
	pub b: f32,
	pub a: f32,
}

impl Color {
	pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
		Self { r, g, b, a }
	}

	/// Normalize 8-bit channels (0–255) to 0–1.
	pub fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
		Self {
			r: r as f32 / 255.0,
			g: g as f32 / 255.0,
			b: b as f32 / 255.0,
			a: a as f32 / 255.0,
		}
	}
}

impl From<Color> for [f32; 4] {
	fn from(c: Color) -> Self {
		[c.r, c.g, c.b, c.a]
	}
}

impl From<Pixel> for Color {
	fn from(p: Pixel) -> Self {
		Color::from_u8(p.red, p.green, p.blue, p.alpha)
	}
}

/// A point normalized to the layer dimensions (0–1 against width/height).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2 {
	pub x: f32,
	pub y: f32,
}

impl Point2 {
	pub const fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}
}

impl From<Point2> for [f32; 2] {
	fn from(p: Point2) -> Self {
		[p.x, p.y]
	}
}

/// One host parameter value, normalized at snapshot time. `Copy` so the
/// generated `Snapshot` array stays `Copy` (AE pre-render data must be `Copy`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamValue {
	Float(f32),
	Bool(bool),
	Color(Color),
	Point(Point2),
	/// Popup selection, 0-based on every host.
	Index(u32),
	/// Buttons and `#[custom]` params have no readable value.
	None,
}

impl Default for ParamValue {
	fn default() -> Self {
		ParamValue::None
	}
}
