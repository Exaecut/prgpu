use prgpu::gpu_struct;

#[gpu_struct]
pub struct WithF32x3 {
    pub rgb: [f32; 3],
}

fn main() {}
