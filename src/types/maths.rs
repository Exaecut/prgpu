use crate::types::Pixel;
use premiere as pr;

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    _pad: u32,
}

impl From<Pixel> for Vec3 {
    fn from(value: Pixel) -> Self {
        Vec3 {
            x: value.red as f32 / 255.0,
            y: value.green as f32 / 255.0,
            z: value.blue as f32 / 255.0,
            _pad: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Transform {
    pub position: Vec2,
    pub scale: Vec2,
    pub angle: f32,
    _pad: u32,
}

pub struct PrRect(pr::sys::prRect);

impl From<pr::sys::prRect> for PrRect {
	fn from(rect: pr::sys::prRect) -> Self {
		PrRect(rect)
	}
}

impl From<PrRect> for after_effects::Rect {
	fn from(rect: PrRect) -> Self {
		after_effects::Rect {
			bottom: rect.0.bottom,
			left: rect.0.left,
			right: rect.0.right,
			top: rect.0.top,
		}
	}
}
