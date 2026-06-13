//! Kernel descriptors and dispatch. Use [`kernel!`](crate::kernel!) in
//! effects; [`Kernel`] is consumed by the graph executor.

mod descriptor;
pub mod params;
pub use descriptor::Kernel;
pub use params::KernelParams;

pub mod builtin;

mod macros;

mod from_ctx;
pub use from_ctx::FromCtx;
