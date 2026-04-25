use prgpu::gpu_struct;

#[gpu_struct]
pub struct WithUsize {
    pub len: usize,
}

fn main() {}
