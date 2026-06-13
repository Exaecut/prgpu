//! Declarative render-graph layer.
//!
//! `Graph<P>` lets an effect describe its multi-pass pipeline once in terms
//! of resources (mip pyramids), single passes, and mip-chain sweeps. The
//! executor runs the graph against a per-frame `Ctx<P>`.

pub mod execute;
pub mod pass;
pub mod resource;
pub mod source;

mod builder;

pub use execute::GraphError;
pub use builder::{Derived, Graph};
pub use pass::{MipDirection, PyramidHandle, Slot};
pub use resource::{MipPyramid, MipPyramidDesc};
pub use source::SourcePolicy;
