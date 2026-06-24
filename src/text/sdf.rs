//! Single-channel signed distance field from a coverage bitmap via 8SSEDT
//! (8-point Sequential Signed Euclidean Distance Transform).
//!
//! The transform runs twice - once seeded on inside pixels, once on outside -
//! and the per-pixel signed distance is `outside_dist - inside_dist`, encoded
//! to u8 with the glyph edge at 0.5. `spread` is the distance in pixels that
//! maps to the full [0,1] range, so the shader recovers crisp edges with a
//! single `smoothstep` regardless of draw scale.

/// Offset from a cell to its nearest seed. `INF` marks "no seed reached yet".
#[derive(Clone, Copy)]
struct Cell {
	dx: i32,
	dy: i32,
}

const INF: Cell = Cell { dx: 16384, dy: 16384 };
const ZERO: Cell = Cell { dx: 0, dy: 0 };

impl Cell {
	#[inline]
	fn dist_sq(self) -> i64 {
		(self.dx as i64) * (self.dx as i64) + (self.dy as i64) * (self.dy as i64)
	}
}

struct Grid {
	w: usize,
	h: usize,
	cells: Vec<Cell>,
}

impl Grid {
	/// Seed cells where `seed(x,y)` holds with offset 0; everything else INF.
	fn seeded(w: usize, h: usize, seed: impl Fn(usize, usize) -> bool) -> Self {
		let mut cells = vec![INF; w * h];
		for y in 0..h {
			for x in 0..w {
				if seed(x, y) {
					cells[y * w + x] = ZERO;
				}
			}
		}
		Self { w, h, cells }
	}

	#[inline]
	fn get(&self, x: i32, y: i32) -> Cell {
		if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
			return INF;
		}
		self.cells[y as usize * self.w + x as usize]
	}

	/// Relax cell (x,y) against the neighbour at (x+ox, y+oy): adopt the
	/// neighbour's seed (shifted by the step) when it yields a shorter distance.
	#[inline]
	fn relax(&mut self, x: usize, y: usize, ox: i32, oy: i32) {
		let mut other = self.get(x as i32 + ox, y as i32 + oy);
		other.dx += ox;
		other.dy += oy;
		let idx = y * self.w + x;
		if other.dist_sq() < self.cells[idx].dist_sq() {
			self.cells[idx] = other;
		}
	}

	/// Two-pass propagation: each cell ends holding the offset to its nearest seed.
	fn propagate(&mut self) {
		for y in 0..self.h {
			for x in 0..self.w {
				self.relax(x, y, -1, 0);
				self.relax(x, y, 0, -1);
				self.relax(x, y, -1, -1);
				self.relax(x, y, 1, -1);
			}
			for x in (0..self.w).rev() {
				self.relax(x, y, 1, 0);
			}
		}
		for y in (0..self.h).rev() {
			for x in (0..self.w).rev() {
				self.relax(x, y, 1, 0);
				self.relax(x, y, 0, 1);
				self.relax(x, y, 1, 1);
				self.relax(x, y, -1, 1);
			}
			for x in 0..self.w {
				self.relax(x, y, -1, 0);
			}
		}
	}
}

/// Build an 8-bit SDF from an alpha-coverage bitmap. `coverage[y*w+x]` is the
/// glyph alpha (0..=255); pixels ≥ 128 are treated as inside. The output byte
/// is `0.5 + signed_dist / (2*spread)` clamped to [0,1] then scaled to 0..=255,
/// so 128 is the edge and `spread` pixels reach the extremes.
pub fn coverage_to_sdf(coverage: &[u8], w: usize, h: usize, spread: f32) -> Vec<u8> {
	if w == 0 || h == 0 {
		return Vec::new();
	}
	let inside = |x: usize, y: usize| coverage[y * w + x] >= 128;

	let mut outside_grid = Grid::seeded(w, h, |x, y| !inside(x, y));
	let mut inside_grid = Grid::seeded(w, h, |x, y| inside(x, y));
	outside_grid.propagate();
	inside_grid.propagate();

	let spread = spread.max(1.0);
	let mut out = vec![0u8; w * h];
	for i in 0..w * h {
		// Distance to the opposite region: inside cells measure to outside, and
		// vice versa. Signed so inside is positive.
		let d_out = (outside_grid.cells[i].dist_sq() as f64).sqrt() as f32;
		let d_in = (inside_grid.cells[i].dist_sq() as f64).sqrt() as f32;
		let signed = if coverage[i] >= 128 { d_out } else { -d_in };
		let norm = 0.5 + signed / (2.0 * spread);
		out[i] = (norm.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	// A filled disc must yield a monotonic SDF: centre saturates high, far
	// corners saturate low, and the radius sits near the 128 edge.
	#[test]
	fn disc_sdf_is_monotonic_about_the_edge() {
		let (w, h) = (64usize, 64usize);
		let (cx, cy, r) = (32.0f32, 32.0f32, 16.0f32);
		let mut cov = vec![0u8; w * h];
		for y in 0..h {
			for x in 0..w {
				let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
				cov[y * w + x] = if d <= r { 255 } else { 0 };
			}
		}
		let sdf = coverage_to_sdf(&cov, w, h, 8.0);

		let centre = sdf[cy as usize * w + cx as usize];
		let corner = sdf[0];
		assert!(centre > 200, "centre should be deep inside, got {centre}");
		assert!(corner < 55, "corner should be deep outside, got {corner}");

		// A texel straddling the radius is within a band of the 128 edge.
		let edge = sdf[cy as usize * w + (cx + r) as usize];
		assert!((edge as i32 - 128).abs() <= 40, "edge texel should be near 128, got {edge}");
	}

	#[test]
	fn empty_input_is_empty() {
		assert!(coverage_to_sdf(&[], 0, 0, 8.0).is_empty());
	}
}
