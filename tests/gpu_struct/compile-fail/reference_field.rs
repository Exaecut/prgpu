use prgpu::gpu_struct;

#[gpu_struct]
pub struct WithRef {
    pub x: &'static f32,
}

fn main() {}
