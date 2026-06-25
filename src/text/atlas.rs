//! Build a single-channel SDF glyph atlas from a TTF with fontdue.
//!
//! Glyphs in the printable-ASCII range are rasterised once at `base_px`, padded
//! by `spread` so the distance field has room to ramp, converted to an SDF, and
//! shelf-packed into one atlas image. The per-glyph [`GlyphMetric`] records both
//! the atlas cell and the pen-relative placement (all in base-px units) the GPU
//! `DrawString` needs to lay text out at any draw size.

use fontdue::{Font, FontSettings};

use crate::text::sdf::coverage_to_sdf;

/// First printable ASCII code point covered by the atlas (space).
pub const FIRST_CHAR: u32 = 32;
/// Last printable ASCII code point covered by the atlas (`~`).
pub const LAST_CHAR: u32 = 126;
/// Number of glyph cells in the atlas (`FIRST_CHAR..=LAST_CHAR`).
pub const GLYPH_COUNT: usize = (LAST_CHAR - FIRST_CHAR + 1) as usize;

/// Atlas image row width in pixels. A power of two keeps GPU word addressing
/// (4 SDF bytes per `uint`) aligned without per-row padding.
const ATLAS_WIDTH: u32 = 512;

/// Per-glyph layout record. All offsets/sizes are in **base-px** units (the size
/// the atlas was rasterised at); the GPU scales by `pxSize / base_px`. Atlas
/// coordinates are in atlas pixels. `#[repr(C)]`, 8×f32 = 32 bytes, uploaded
/// verbatim as the GPU metrics buffer — keep byte-identical with the Slang
/// `GlyphMetric`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphMetric {
	/// Atlas-pixel top-left of the glyph's SDF cell.
	pub atlas_x: f32,
	pub atlas_y: f32,
	/// SDF cell size in atlas pixels (= base-px cell size, includes 2×spread).
	pub cell_w: f32,
	pub cell_h: f32,
	/// Cell top-left relative to the pen origin on the baseline (y grows down).
	pub left: f32,
	pub top: f32,
	/// Pen advance to the next glyph.
	pub advance: f32,
	pub _pad: f32,
}

/// A built SDF atlas plus the metrics and font geometry needed to render it.
pub struct Atlas {
	pub width: u32,
	pub height: u32,
	/// Single-channel SDF, `width * height` bytes, edge at 128.
	pub pixels: Vec<u8>,
	/// One entry per `FIRST_CHAR..=LAST_CHAR`, indexed by `code - FIRST_CHAR`.
	pub metrics: Vec<GlyphMetric>,
	pub base_px: f32,
	pub spread: f32,
	/// Baseline-to-baseline distance in base-px.
	pub line_height: f32,
	/// Baseline-to-top (ascent) in base-px, positive.
	pub ascent: f32,
	/// Baseline-to-bottom (descent) in base-px, negative.
	pub descent: f32,
}

/// The font bytes baked into prgpu (Roboto Regular, Apache-2.0).
pub const EMBEDDED_FONT: &[u8] = include_bytes!("assets/font.ttf");

/// Build the default atlas from the embedded font at a 48-px base with a 6-px
/// SDF spread — a good balance of atlas size and edge quality for UI overlays.
pub fn build_default_atlas() -> Atlas {
	build_atlas(EMBEDDED_FONT, 48.0, 6.0)
}

/// Rasterise the printable-ASCII glyphs of `ttf` to an SDF atlas. `spread` is
/// the SDF ramp half-width in base-px (also the inter-glyph padding).
pub fn build_atlas(ttf: &[u8], base_px: f32, spread: f32) -> Atlas {
	let font = Font::from_bytes(ttf, FontSettings { scale: base_px, ..FontSettings::default() }).expect("prgpu::text: embedded font failed to parse");

	let line = font.horizontal_line_metrics(base_px).unwrap_or(fontdue::LineMetrics {
		ascent: base_px * 0.8,
		descent: -base_px * 0.2,
		line_gap: 0.0,
		new_line_size: base_px,
	});

	let pad = spread.ceil() as i32;
	let mut metrics = vec![GlyphMetric::default(); GLYPH_COUNT];

	// First pass: rasterise + SDF each glyph, recording its cell pixels and the
	// pen-relative placement. Packing happens after so we know every cell size.
	struct Cell {
		idx: usize,
		w: u32,
		h: u32,
		sdf: Vec<u8>,
	}
	let mut cells: Vec<Cell> = Vec::with_capacity(GLYPH_COUNT);

	for code in FIRST_CHAR..=LAST_CHAR {
		let idx = (code - FIRST_CHAR) as usize;
		let ch = char::from_u32(code).unwrap();
		let (m, bitmap) = font.rasterize(ch, base_px);

		// Whitespace (and any empty glyph) has no pixels: record advance only.
		if m.width == 0 || m.height == 0 {
			metrics[idx] = GlyphMetric { advance: m.advance_width, ..Default::default() };
			continue;
		}

		let cw = m.width as i32 + 2 * pad;
		let ch_px = m.height as i32 + 2 * pad;
		let mut cov = vec![0u8; (cw * ch_px) as usize];
		for y in 0..m.height {
			for x in 0..m.width {
				cov[((y as i32 + pad) * cw + x as i32 + pad) as usize] = bitmap[y * m.width + x];
			}
		}
		let sdf = coverage_to_sdf(&cov, cw as usize, ch_px as usize, spread);

		// fontdue: xmin = left bearing, ymin = baseline→bitmap-bottom (up +).
		// Cell top-left (y down) sits `pad` above/left of the glyph bitmap.
		let left = m.xmin as f32 - pad as f32;
		let top = -(m.ymin as f32 + m.height as f32) - pad as f32;

		metrics[idx] = GlyphMetric {
			cell_w: cw as f32,
			cell_h: ch_px as f32,
			left,
			top,
			advance: m.advance_width,
			..Default::default()
		};
		cells.push(Cell { idx, w: cw as u32, h: ch_px as u32, sdf });
	}

	// Shelf pack the cells into ATLAS_WIDTH rows.
	let mut cursor_x = 0u32;
	let mut cursor_y = 0u32;
	let mut shelf_h = 0u32;
	for c in &cells {
		if cursor_x + c.w > ATLAS_WIDTH {
			cursor_y += shelf_h;
			cursor_x = 0;
			shelf_h = 0;
		}
		metrics[c.idx].atlas_x = cursor_x as f32;
		metrics[c.idx].atlas_y = cursor_y as f32;
		cursor_x += c.w;
		shelf_h = shelf_h.max(c.h);
	}
	let height = (cursor_y + shelf_h).max(1);

	// Blit each cell's SDF into the packed atlas.
	let mut pixels = vec![0u8; (ATLAS_WIDTH * height) as usize];
	for c in &cells {
		let ax = metrics[c.idx].atlas_x as u32;
		let ay = metrics[c.idx].atlas_y as u32;
		for row in 0..c.h {
			let dst = ((ay + row) * ATLAS_WIDTH + ax) as usize;
			let src = (row * c.w) as usize;
			pixels[dst..dst + c.w as usize].copy_from_slice(&c.sdf[src..src + c.w as usize]);
		}
	}

	Atlas {
		width: ATLAS_WIDTH,
		height,
		pixels,
		metrics,
		base_px,
		spread,
		line_height: line.new_line_size,
		ascent: line.ascent,
		descent: line.descent,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn default_atlas_covers_ascii_and_packs() {
		let atlas = build_default_atlas();
		assert_eq!(atlas.metrics.len(), GLYPH_COUNT);
		assert_eq!(atlas.width, ATLAS_WIDTH);
		assert!(atlas.height > 0);
		assert_eq!(atlas.pixels.len(), (atlas.width * atlas.height) as usize);

		// 'A' (code 65) must have a non-empty cell within atlas bounds and a
		// positive advance; space (32) advances but has no cell.
		let a = atlas.metrics[(b'A' as u32 - FIRST_CHAR) as usize];
		assert!(a.cell_w > 0.0 && a.cell_h > 0.0, "'A' should have a cell");
		assert!(a.advance > 0.0);
		assert!(a.atlas_x + a.cell_w <= atlas.width as f32);
		assert!(a.atlas_y + a.cell_h <= atlas.height as f32);

		let space = atlas.metrics[0];
		assert!(space.advance > 0.0, "space must advance");
		assert_eq!(space.cell_w, 0.0, "space must have no cell");
	}

	#[test]
	fn glyph_metric_is_32_bytes() {
		assert_eq!(core::mem::size_of::<GlyphMetric>(), 32);
	}
}
