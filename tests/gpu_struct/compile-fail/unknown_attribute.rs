use prgpu::gpu_struct;

#[gpu_struct(unknown_attr)]
pub struct BadAttr {
    pub x: f32,
}

fn main() {}
