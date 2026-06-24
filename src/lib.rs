//! # prgpu — declarative GPU effect framework for Adobe host plug-ins
//!
//! ```ignore
//! // playground/src/lib.rs — a complete effect in ~45 lines
//! use prgpu::prelude::*;
//! pub mod kernel; pub mod params;
//! use crate::{kernel::playground, params::*};
//! pub struct Playground;
//! impl Effect for Playground {
//!     type Params = Params;
//!     fn descriptor(d: EffectDescriptor) -> EffectDescriptor { d.about("…").options_button("…") }
//!     fn expansion(ctx: &Ctx<Params>) -> ExpansionExtent { /* ... */ }
//!     fn pipeline(g: &mut Graph<Params>) { g.pass(playground::kernel()); }
//! }
//! prgpu::register_effect!(Playground);
//! ```
//!
//! Concept map: `params!` → `kernel!` → `Effect` → `register_effect!` → `build()`.

// Lets the `Popup` derive and `params!` codegen reference `::prgpu::*` even when
// expanded inside this crate (e.g. on `BlendMode`).
extern crate self as prgpu;

pub mod prelude;
pub use prelude::*;

mod pipeline;

pub mod kernel;
pub mod graph;
pub mod adobe;
pub mod effect;
pub mod params;
pub mod types;
pub mod cpu;
pub mod gpu;
pub mod text;
pub mod timing;

pub use paste;
pub use prgpu_macro::{Popup, gpu_struct, kernel, params};

mod register_effect;

#[cfg(feature = "bench")]
pub mod bench;

#[cfg(feature = "testing")]
pub mod testing;
