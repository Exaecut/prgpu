//! Built-in text-overlay constant buffer.
//!
//! Byte-identical to the Slang `vekl::TextDrawParams` (tight 4-byte scalar
//! layout, `color`/`bg_color`/`packed` are arrays — not vectors — so no 16-byte
//! vec alignment). 96-byte header + 256-byte packed char block = 352 bytes, a
//! multiple of 16. `packed` stores up to 256 char codes, four per word.

use crate::kernel::params::KernelParams;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TextOverlayParams {
	pub color: [f32; 4],
	/// Full-width background band colour (straight RGBA). Alpha 0 = no band.
	pub bg_color: [f32; 4],
	pub pen_x: f32,
	pub pen_y: f32,
	pub scale: f32,
	pub spread: f32,
	pub atlas_w: u32,
	pub atlas_h: u32,
	pub frame_w: u32,
	pub frame_h: u32,
	pub bbox_x: u32,
	pub bbox_y: u32,
	pub bbox_w: u32,
	pub bbox_h: u32,
	pub char_count: u32,
	pub first_char: u32,
	pub glyph_count: u32,
	pub _pad0: u32,
	pub packed: [u32; 64],
}

impl KernelParams for TextOverlayParams {
	const SIZE: usize = core::mem::size_of::<Self>();
	const ALIGN: usize = core::mem::align_of::<Self>();
}

// The Slang TextDrawParams is a tight 4-byte-scalar layout; mirror its size.
const _: () = assert!(core::mem::size_of::<TextOverlayParams>() == 352);
