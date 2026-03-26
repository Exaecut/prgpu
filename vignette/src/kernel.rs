use prgpu::declare_kernel;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VignetteParams {}

declare_kernel!(vignette, VignetteParams);
