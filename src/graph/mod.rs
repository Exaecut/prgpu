//! Declarative multi-pass graph. [`Graph::pass`](Graph::pass) declares a
//! kernel invocation; [`Graph::mip_chain`](Graph::mip_chain) sweeps a pyramid.

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
