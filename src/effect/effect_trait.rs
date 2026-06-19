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

	/// Opt into the background-task idle pump. When `true`, the adapter
	/// registers an AE idle hook at GlobalSetup that drives
	/// [`prgpu::tasks`](crate::effect::tasks) on the main thread. Leave `false`
	/// for effects that never spawn background work.
	const USES_BACKGROUND_TASKS: bool = false;

	/// Override points on the build-metadata descriptor (match name, version,
	/// flags come from `[package.metadata.prgpu]` via `register_effect!`).
	fn descriptor(_d: EffectDescriptor) -> EffectDescriptor {
		_d
	}

	/// Raw SDK access for cases `params!` doesn't cover.
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

	/// Custom-UI event hook. Called by the AE adapter on `PF_Cmd_EVENT` for
	/// params with `PF_PUI_CONTROL` (e.g. `#[label]`). Default no-op.
	///
	/// Label params (`P::LABEL_PARAMS`) are drawn automatically by the adapter
	/// via Drawbot `draw_string` using the text stashed by `Ui::set_label`;
	/// override this only if you need fully custom drawing.
	fn on_event(
		_in_data: &after_effects::InData,
		_params: &mut after_effects::Parameters<Self::Params>,
		_event: &mut after_effects::EventExtra,
	) -> Result<(), after_effects::Error> {
		Ok(())
	}

	/// Low-level escape hatch. Called at the top of every PF command dispatch,
	/// before prgpu's own handling, with the raw Adobe SDK [`Command`] plus
	/// `InData`/`OutData`/`Parameters`. Use it to hook host-specific commands or
	/// events the declarative API doesn't model — e.g. acquiring a private
	/// Premiere host suite at `UpdateParamsUi`/`SequenceSetup`. Default no-op;
	/// prgpu always proceeds with its normal handling afterwards.
	///
	/// [`Command`]: after_effects::Command
	fn on_raw_command(
		_command: &after_effects::Command,
		_in_data: &after_effects::InData,
		_out_data: &mut after_effects::OutData,
		_params: &mut after_effects::Parameters<Self::Params>,
	) -> Result<(), after_effects::Error> {
		Ok(())
	}
}
