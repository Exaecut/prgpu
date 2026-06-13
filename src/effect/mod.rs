//! Effect authoring surface: implement [`Effect`], call
//! [`register_effect!`](crate::register_effect), done.

pub mod host;
pub use host::{Capability, Host, HostCapabilities, RenderKind};

pub mod invocation;
pub use invocation::{FrameBinding, InvocationBase, PixelLayout};

pub mod license;
pub use license::{LicenseGate, NoLicense};

pub mod descriptor;
pub use descriptor::{EffectDescriptor, ExpansionExtent};

pub mod ctx;
pub use ctx::{Ctx, Geometry, Timing};

pub mod effect_trait;
pub use effect_trait::Effect;

pub mod ui;
pub use ui::Ui;

pub mod prelude;
pub use prelude::*;

pub mod meta;
pub use meta::EffectMeta;
