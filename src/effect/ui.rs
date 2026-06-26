//! Snapshot-driven visibility + label rules. `Effect::ui` collects rules
//! once; the adapter evaluates them against each frame's `Ctx<P>`.
//!
//! Replaces `ParamApi`/`VisibilityBuilder`/`ActionBuilder` — action callbacks
//! live on `params!` `#[button(on_click = ...)]` attributes (phase 3).

use crate::effect::Ctx;
use crate::params::{Param, ParamsSpec};

pub struct Ui<P: ParamsSpec> {
	/// Each rule: (param, predicate). Predicate is evaluated against `Ctx<P>`.
	pub(crate) rules: Vec<(P, Box<dyn Fn(&Ctx<P>) -> bool + Send + Sync + 'static>)>,
	/// Each label rule: (param, label_fn). Returns the new label text for the
	/// param's UI row, evaluated on every UpdateParamsUi.
	pub(crate) label_rules: Vec<(P, Box<dyn Fn(&Ctx<P>) -> String + Send + Sync + 'static>)>,
	/// Each disable rule: (param, predicate). Gray-out the param via
	/// `PF_PUI_DISABLED` while the predicate returns `true`, same cadence as
	/// `rules` / `label_rules`.
	pub(crate) disabled_rules: Vec<(P, Box<dyn Fn(&Ctx<P>) -> bool + Send + Sync + 'static>)>,
	/// Each color rule: (param, color_fn) → RGBA 0..1 for a `#[label]` row's
	/// text, evaluated per UI tick like `label_rules`.
	pub(crate) color_rules: Vec<(P, Box<dyn Fn(&Ctx<P>) -> [f32; 4] + Send + Sync + 'static>)>,
}

impl<P: ParamsSpec> Ui<P> {
	pub fn new() -> Self {
		Self { rules: Vec::new(), label_rules: Vec::new(), disabled_rules: Vec::new(), color_rules: Vec::new() }
	}

	pub fn show<M: Param<Spec = P>>(
		&mut self,
		_m: M,
		pred: impl Fn(&Ctx<P>) -> bool + Send + Sync + 'static,
	) {
		self.rules.push((M::ID, Box::new(pred)));
	}

	pub fn show_all<M: Param<Spec = P>, const N: usize>(
		&mut self,
		_ms: [M; N],
		pred: impl Fn(&Ctx<P>) -> bool + Clone + Send + Sync + 'static,
	) {
		for m in _ms {
			self.rules.push((M::ID, Box::new(pred.clone())));
		}
	}

	/// Dynamically relabel a param's UI row. The closure receives the current
	/// `Ctx<P>` and returns the new label. Evaluated on every UpdateParamsUi /
	/// UserChangedParam, same cadence as `show`. For ctx-independent text,
	/// prefer the declarative `#[label(text = …)]` form in `params!`.
	pub fn set_label<M: Param<Spec = P>>(
		&mut self,
		_m: M,
		label_fn: impl Fn(&Ctx<P>) -> String + Send + Sync + 'static,
	) {
		self.label_rules.push((M::ID, Box::new(label_fn)));
	}

	/// Marker-free variant of [`set_label`](Self::set_label), used by the
	/// macro-generated `#[label(text = …)]` bindings.
	#[doc(hidden)]
	pub fn set_label_id(
		&mut self,
		id: P,
		label_fn: impl Fn(&Ctx<P>) -> String + Send + Sync + 'static,
	) {
		self.label_rules.push((id, Box::new(label_fn)));
	}

	/// Toggles `PF_PUI_DISABLED` each UI tick; the param is grayed-out while
	/// the predicate returns `true`. Ctx-dependent counterpart of the
	/// declarative `#[button(disabled = …)]` form; mirrors [`set_label`](Self::set_label).
	pub fn set_disabled<M: Param<Spec = P>>(
		&mut self,
		_m: M,
		pred: impl Fn(&Ctx<P>) -> bool + Send + Sync + 'static,
	) {
		self.disabled_rules.push((M::ID, Box::new(pred)));
	}

	/// Set a `#[label]` row's text color (RGBA 0..1), evaluated per UI tick like
	/// [`set_label`](Self::set_label). Only custom-draw `#[label]` params honor
	/// it; name-driven params (buttons/popups) ignore it.
	pub fn set_label_color<M: Param<Spec = P>>(
		&mut self,
		_m: M,
		color_fn: impl Fn(&Ctx<P>) -> [f32; 4] + Send + Sync + 'static,
	) {
		self.color_rules.push((M::ID, Box::new(color_fn)));
	}

	/// Marker-free variant of [`set_disabled`](Self::set_disabled), used by the
	/// macro-generated `#[button(disabled = …)]` bindings.
	#[doc(hidden)]
	pub fn set_disabled_id(
		&mut self,
		id: P,
		pred: impl Fn(&Ctx<P>) -> bool + Send + Sync + 'static,
	) {
		self.disabled_rules.push((id, Box::new(pred)));
	}
}

impl<P: ParamsSpec> Default for Ui<P> {
	fn default() -> Self {
		Self::new()
	}
}
