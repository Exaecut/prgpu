use prgpu::gpu_struct;

#[gpu_struct]
pub struct WithTuple {
    pub pair: (f32, f32),
}

fn main() {}
