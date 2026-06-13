//! Parameter declaration, normalization, and typed read surface.
//!
//! `params!` is the single source of truth for an effect's parameters: it
//! generates the discriminant enum, one zero-sized [`Param`] marker per
//! variant, the [`ParamsSpec`] (host registration + per-frame [`Snapshot`]),
//! and a legacy [`SetupParams`] bridge. `Ctx::get(Marker)` is the one read API.

pub const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

pub mod blend;
pub mod convert;
pub mod legacy;
pub mod traits;
pub mod value;

pub use blend::BlendMode;
pub use traits::{FromParamValue, Param, ParamsSpec, PopupOptions, Snapshot, SnapshotGeom};
pub use value::{Color, ParamValue, Point2};

// TRANSITIONAL(plan-05): re-exported so `kernel_params!`, `FrameDataContext`,
// and the not-yet-ported effect bodies keep resolving `prgpu::params::*`.
pub use legacy::{CpuParams, FromParam, SetupParams, get_param, register_gpu_param_indices};
