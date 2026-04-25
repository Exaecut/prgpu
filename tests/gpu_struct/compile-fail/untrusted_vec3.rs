use prgpu::gpu_struct;

mod fake { pub struct Vec3; }

#[gpu_struct]
pub struct UntrustedVec3 {
    pub v: fake::Vec3,
}

fn main() {}
