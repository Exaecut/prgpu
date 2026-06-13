//! Parameter declaration via [`params!`](crate::params!) and typed read
//! surface. [`Ctx::get`](crate::Ctx::get) is the one read API.

pub const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

pub mod blend;
pub mod convert;
pub mod legacy;
pub mod traits;
pub mod value;

pub use blend::BlendMode;
pub use traits::{FromParamValue, Param, ParamsSpec, PopupOptions, Snapshot, SnapshotGeom};
pub use value::{Color, ParamValue, Point2};

pub use legacy::{CpuParams, FromParam, SetupParams, get_param, register_gpu_param_indices};
