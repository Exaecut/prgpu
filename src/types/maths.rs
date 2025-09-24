use crate::types::Pixel;

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, bytemuck::Zeroable)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Pixel> for Vec3 {
    fn from(value: Pixel) -> Self {
        Vec3 {
            x: value.red as f32 / 255.0,
            y: value.green as f32 / 255.0,
            z: value.blue as f32 / 255.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Zeroable)]
pub struct Transform {
    pub position: Vec2,
    pub scale: Vec2,
    pub angle: f32,
}
