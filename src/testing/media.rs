//! Deterministic BGRA test images.

#[derive(Clone, Copy, Debug)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    pub const BLACK: Self = Self { r: 0, g: 0, b: 0, a: 255 };
    pub const WHITE: Self = Self { r: 255, g: 255, b: 255, a: 255 };
    pub const RED: Self = Self { r: 255, g: 0, b: 0, a: 255 };
    pub const GREEN: Self = Self { r: 0, g: 255, b: 0, a: 255 };
    pub const BLUE: Self = Self { r: 0, g: 0, b: 255, a: 255 };
    pub const MAGENTA: Self = Self { r: 255, g: 0, b: 255, a: 255 };
    pub const TRANSPARENT: Self = Self { r: 0, g: 0, b: 0, a: 0 };

    /// BGRA byte order — the GPU path convention.
    pub fn to_bgra(self) -> [u8; 4] {
        [self.b, self.g, self.r, self.a]
    }
}

pub type BgraData = Vec<u8>;

/// Tile size = 32 px. Returns `width × height × 4` BGRA bytes.
pub fn builtin_checkerboard(width: u32, height: u32) -> BgraData {
    let tile = 32u32;
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];
    for y in 0..height {
        for x in 0..width {
            let off = ((y * width + x) * 4) as usize;
            let is_white = ((x / tile) + (y / tile)) % 2 == 0;
            let px = if is_white { Rgba8::WHITE } else { Rgba8::BLACK };
            let bgra = px.to_bgra();
            data[off..off + 4].copy_from_slice(&bgra);
        }
    }
    data
}

pub fn builtin_solid_color(width: u32, height: u32, color: Rgba8) -> BgraData {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];
    let bgra = color.to_bgra();
    for chunk in data.chunks_exact_mut(4) {
        chunk.copy_from_slice(&bgra);
    }
    data
}

pub fn builtin_gradient_h(width: u32, height: u32, left: Rgba8, right: Rgba8) -> BgraData {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];
    let wf = width as f32;
    for y in 0..height {
        for x in 0..width {
            let t = x as f32 / (wf - 1.0).max(1.0);
            let r = lerp_u8(left.r, right.r, t);
            let g = lerp_u8(left.g, right.g, t);
            let b = lerp_u8(left.b, right.b, t);
            let a = lerp_u8(left.a, right.a, t);
            let off = ((y * width + x) * 4) as usize;
            data[off] = b;
            data[off + 1] = g;
            data[off + 2] = r;
            data[off + 3] = a;
        }
    }
    data
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

/// Loads a PNG from disk and converts RGBA → BGRA. Returns `(data, width, height)`.
pub fn load_png_bgra8(path: &str) -> Result<(BgraData, u32, u32), String> {
    let img = image::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let mut bgra = rgba.into_raw();
    for chunk in bgra.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }
    Ok((bgra, w, h))
}
