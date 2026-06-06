//! Effect-side public API.
//!
//! Two layers live here:
//!
//! - [`cross_host`] — the existing `CrossHostEffect<P>` AE PF dispatcher
//!   trait. Effects that haven't migrated to the new graph API still
//!   implement this and call its `handle_*` helpers from `handle_command`.
//! - [`host`] / [`invocation`] — the normalised host-capability + invocation
//!   surface the adapters and graph executor build on. Effect code that
//!   uses the new `Effect` trait sees these through `FrameDataContext` /
//!   `ExpansionContext` rather than directly.

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
