//! Declarative render-graph layer.
//!
//! `RenderGraph<F>` lets an effect describe its multi-pass pipeline once in
//! terms of resources (mip pyramids, ...), passes (single source→target
//! kernel invocations), and mip-chain sweeps (down/up across an N-level
//! pyramid). The executor then runs the graph against a per-frame
//! [`crate::effect::InvocationBase`] without the effect having to assemble
//! per-pass `Configuration` values by hand.
//!
//! `F` is the effect's `FrameData` — a `Copy + 'static` struct the per-pass
//! params closures pull values out of. The `Effect` trait parameterises the
//! graph over `Effect::FrameData` directly.

pub mod context;
pub mod execute;
pub mod pass;
pub mod resource;
pub mod source;

mod builder;

pub use builder::{GraphError, RenderGraph};
pub use context::{MipPyramidCtx, PassContext};
pub use pass::{MipDirection, Slot};
pub use resource::{MipPyramid, MipPyramidDesc, ResourceHandle, ResourceLifetime};
pub use source::SourcePolicy;
