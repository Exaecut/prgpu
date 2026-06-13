//! The complete, intentional public surface. Effect crates should use
//! `use prgpu::prelude::*;` as their single import.

pub use crate::effect::{
    Capability, Ctx, Effect, EffectDescriptor, ExpansionExtent, LicenseGate, Ui,
};
pub use crate::graph::{
    Derived, Graph, MipDirection, MipPyramidDesc, PyramidHandle, Slot, SourcePolicy,
};
pub use crate::kernel::{FromCtx, Kernel, KernelParams};
pub use crate::params::{
    BlendMode, Color, FromParamValue, Param, ParamValue, ParamsSpec, Point2, PopupOptions,
    Snapshot, SnapshotGeom, DEG_TO_RAD,
};
pub use crate::types::Backend;
