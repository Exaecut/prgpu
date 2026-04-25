use prgpu::gpu_struct;

#[gpu_struct]
pub struct Generic<T> {
    pub x: T,
}

fn main() {}
