//! Effect-side public API.
//!
//! Effect authors implement [`Effect`] once and let the adapters (AE / Premiere)
//! handle host-specific wiring. The prelude collects the symbols needed for a
//! typical effect definition; the submodules expose the normalised host-capability
//! and invocation surface the adapters and graph executor build on.

pub mod host;
pub use host::{Capability, Host, HostCapabilities, RenderKind};

pub mod invocation;
pub use invocation::{FrameBinding, InvocationBase, PixelLayout};

pub mod params_api;
pub use params_api::{ActionBuilder, ActionContext, ParamApi, VisibilityBuilder};

pub mod license;
pub use license::{LicenseGate, NoLicenseGate};

pub mod descriptor;
pub use descriptor::{EffectDescriptor, ExpansionExtent};

pub mod frame_context;
pub use frame_context::{ExpansionContext, FrameDataContext};

pub mod effect_trait;
pub use effect_trait::Effect;

pub mod prelude;
pub use prelude::*;

pub mod meta;
pub use meta::EffectMeta;
