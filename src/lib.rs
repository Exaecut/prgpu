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
pub mod ui;

#[cfg(feature = "build")]
pub mod build;

#[cfg(feature = "bench")]
pub mod bench;

#[cfg(feature = "testing")]
pub mod testing;

pub use paste;
pub use prgpu_macro::gpu_struct;
