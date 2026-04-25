use prgpu::gpu_struct;

#[gpu_struct]
#[repr(packed)]
pub struct Packed {
    pub x: u32,
    pub y: f32,
}

fn main() {}
