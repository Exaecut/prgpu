// Lets the `Popup` derive and `params!` codegen reference `::prgpu::*` even when
// expanded inside this crate (e.g. on `BlendMode`).
extern crate self as prgpu;

pub mod kernel;
pub use kernel::{Kernel, KernelParams};

mod pipeline;

pub mod graph;
pub mod adobe;

pub mod cpu;
pub mod gpu;

pub mod timing;

pub mod types;
pub use types::{Backend, ConfigBuildError, ConfigBuilder, Configuration, FrameScopeDesc, PassBinding};

pub mod effect;
pub use effect::prelude::*;

pub mod params;
pub use params::{
	BlendMode, Color, DEG_TO_RAD, FromParamValue, Param, ParamValue, ParamsSpec, Point2, PopupOptions, Snapshot, SnapshotGeom,
};

#[cfg(feature = "bench")]
pub mod bench;

#[cfg(feature = "testing")]
pub mod testing;

pub use paste;
pub use prgpu_macro::{Popup, gpu_struct, params};
