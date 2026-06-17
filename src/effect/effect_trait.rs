//! The single public authoring trait.
//!
//! `Effect` is the high-level surface effect crates implement. The adapters
//! drive every selector via these methods so authors stop hand-writing
//! host-specific selectors.

use premiere as pr;

use crate::effect::ctx::Ctx;
use crate::effect::descriptor::{EffectDescriptor, ExpansionExtent};
use crate::effect::ui::Ui;
use crate::graph::Graph;
use crate::params::ParamsSpec;

/// Public effect-author trait, v2.
///
/// FrameData is gone — the snapshot is the frame data. Pipeline closures
/// receive `&Ctx<P>`. License wiring is automatic via `register_effect!`.
pub trait Effect: Sized + Send + Sync + 'static {
	type Params: ParamsSpec;

	/// Override points on the build-metadata descriptor (match name, version,
	/// flags come from `[package.metadata.prgpu]` via `register_effect!`).
	fn descriptor(_d: EffectDescriptor) -> EffectDescriptor {
		_d
	}

	/// Raw SDK access for cases `params!` doesn't cover (Ground rule 9).
	fn extra_params(
		_p: &mut after_effects::Parameters<Self::Params>,
	) -> Result<(), after_effects::Error> {
		Ok(())
	}

	/// Dynamic parameter visibility. Evaluated on every UpdateParamsUi /
	/// UserChangedParam against a fresh snapshot.
	fn ui(_u: &mut Ui<Self::Params>) {}

	/// Per-side output inflation. Geometry in `ctx` is layer-sized; snapshot
	/// values are current.
	fn expansion(_ctx: &Ctx<Self::Params>) -> ExpansionExtent {
		ExpansionExtent::NONE
	}

	/// Declared once per effect lifetime; closures re-evaluate per frame
	/// against each frame's `Ctx<P>`.
	fn pipeline(_g: &mut Graph<Self::Params>);

	/// Per-frame GPU hook with raw filter access. Runs on the host GPU thread
	/// after the snapshot + `Ctx` are built and before graph execution.
	/// This is the only authoring hook where an effect can reach
	/// `pr::GpuFilterData` — `video_segment_suite`, `timeline_id()`,
	/// `node_id()`, `ppix_suite`, `gpu_device_suite` are all live here.
	/// Default no-op. Never called on the CPU path.
	fn on_gpu_frame(_filter: &pr::GpuFilterData, _render_params: &pr::RenderParams, _ctx: &Ctx<Self::Params>) {}
}
