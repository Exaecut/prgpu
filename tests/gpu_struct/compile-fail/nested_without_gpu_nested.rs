use prgpu::gpu_struct;

pub struct Custom {
    pub x: f32,
}

#[gpu_struct]
pub struct WithoutGpuNested {
    pub inner: Custom,
}

fn main() {}
