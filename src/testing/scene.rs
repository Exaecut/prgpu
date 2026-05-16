//! Canvas, layers, transforms, and timeline for a render test.

use crate::testing::media::Rgba8;

#[derive(Clone, Debug)]
pub enum Media {
    Color(Rgba8),
    /// BGRA bytes at the given dimensions.
    Raw {
        data: Vec<u8>,
        width: u32,
        height: u32,
    },
}

impl Media {
    pub fn solid(color: Rgba8) -> Self {
        Media::Color(color)
    }

    pub fn from_bgra(data: Vec<u8>, width: u32, height: u32) -> Self {
        Media::Raw { data, width, height }
    }
}

#[derive(Clone, Debug)]
pub enum Background {
    Color(Rgba8),
    Transparent,
}

#[derive(Clone, Copy, Debug)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Origin = layer centre.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position_px: Vec2,
    /// 1.0 = original size.
    pub scale: f32,
    /// Degrees around the layer centre.
    pub rotation_degrees: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position_px: Vec2::new(0.0, 0.0),
            scale: 1.0,
            rotation_degrees: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub media: Media,
    pub transform: Transform,
    pub opacity: f32, // [0, 1]
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub background: Background,
    pub layers: Vec<Layer>,
}

/// Simulated Premiere timeline (not the real Adobe object).
#[derive(Clone, Copy, Debug)]
pub struct Timeline {
    pub start_frame: u32,
    pub frame_count: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    /// Adobe ticks per second.
    pub time_scale: i64,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            start_frame: 0,
            frame_count: 1,
            fps_num: 60,
            fps_den: 1,
            time_scale: 60_000,
        }
    }
}

impl Timeline {
    /// Adobe ticks at the given 0-based frame index.
    pub fn clip_time(&self, frame_index: u32) -> i64 {
        let seconds_per_frame = self.fps_den as f64 / self.fps_num as f64;
        let start_seconds = self.start_frame as f64 * seconds_per_frame;
        let t = start_seconds + frame_index as f64 * seconds_per_frame;
        (t * self.time_scale as f64).round() as i64
    }

    pub fn ticks_per_frame(&self) -> i64 {
        let seconds_per_frame = self.fps_den as f64 / self.fps_num as f64;
        (seconds_per_frame * self.time_scale as f64).round() as i64
    }
}
