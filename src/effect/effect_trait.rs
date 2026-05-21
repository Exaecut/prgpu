//! The single public authoring trait. See `prgpu/docs/effect_api.md`.
//!
//! `Effect` is the high-level surface effect crates implement. The Adobe
//! adapters ([`crate::adobe::ae::EffectAdapter`],
//! [`crate::adobe::premiere::GpuFilterAdapter`]) drive every selector via
//! these methods so authors stop hand-writing `handle_command` /
//! `pr::GpuFilter::render` per effect.

use std::fmt::Debug;
use std::hash::Hash;

use after_effects::{InData, OutData, Parameters};

use crate::effect::descriptor::{EffectDescriptor, ExpansionExtent};
use crate::effect::frame_context::{ExpansionContext, FrameDataContext};
use crate::effect::license::LicenseGate;
use crate::effect::params_api::ParamApi;
use crate::graph::RenderGraph;
use crate::params::SetupParams;

/// Public effect-author trait.
///
/// `type License` is required (Rust stable does not support associated-type
/// defaults). Effects without a licence check declare
/// `type License = prgpu::effect::NoLicenseGate;` — the licence trait's
/// default implementation always succeeds.
pub trait Effect: Sized + Default + Send + Sync + 'static {
	type Params: SetupParams + Eq + Hash + Copy + Debug + Send + Sync + 'static;
	type FrameData: Copy + Send + Sync + 'static;
	type License: LicenseGate;

	fn descriptor() -> EffectDescriptor;

	/// Adobe parameter setup. Called once during `Cmd_ParamsSetup`.
	fn params(params: &mut Parameters<Self::Params>, in_data: InData, out_data: OutData) -> Result<(), after_effects::Error>;

	/// Visibility + click-action declaration. Called every `Cmd_UpdateParamsUi`
	/// and on every `Cmd_UserChangedParam`. Default is empty (no dynamic UI).
	fn ui(_api: &mut ParamApi<Self::Params>) -> Result<(), after_effects::Error> {
		Ok(())
	}

	/// Build the per-frame `FrameData` from host parameters + dimensions.
	/// Runs once per frame on `Cmd_FrameSetup` (AE) / `Cmd_PreRender` (Pr).
	fn frame_data(ctx: FrameDataContext<Self::Params>) -> Result<Self::FrameData, after_effects::Error>;

	/// Per-side pixel inflation applied uniformly to the input layer to
	/// compute the rendered output rect. Default returns no expansion.
	fn expansion(_ctx: ExpansionContext<Self::Params>) -> Result<ExpansionExtent, after_effects::Error> {
		Ok(ExpansionExtent::none())
	}

	/// Declare the render graph once per effect-instance lifetime. The
	/// adapter caches the resulting `RenderGraph<FrameData>` so closures
	/// re-evaluate against each frame's `FrameData` without rebuilding.
	fn pipeline(graph: &mut RenderGraph<Self::FrameData>);
}
