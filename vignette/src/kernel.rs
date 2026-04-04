use prgpu::declare_kernel;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VignetteParams {
    pub strength: f32,
    pub softness: f32,
}

declare_kernel!(vignette, VignetteParams);
