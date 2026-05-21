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

pub mod cross_host;
pub use cross_host::*;

pub mod host;
pub use host::{Capability, Host, HostCapabilities, RenderKind};

pub mod invocation;
pub use invocation::{FrameBinding, InvocationBase, PixelLayout};
