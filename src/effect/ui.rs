//! Snapshot-driven visibility rules. `Effect::ui` collects rules once; the
//! adapter evaluates them against each frame's `Ctx<P>`.
//!
//! Replaces `ParamApi`/`VisibilityBuilder`/`ActionBuilder` — action callbacks
//! live on `params!` `#[button(on_click = ...)]` attributes (phase 3).

use crate::effect::Ctx;
use crate::params::{Param, ParamsSpec};

pub struct Ui<P: ParamsSpec> {
	/// Each rule: (param, predicate). Predicate is evaluated against `Ctx<P>`.
	pub(crate) rules: Vec<(P, Box<dyn Fn(&Ctx<P>) -> bool + Send + Sync + 'static>)>,
}

impl<P: ParamsSpec> Ui<P> {
	pub fn new() -> Self {
		Self { rules: Vec::new() }
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
}

impl<P: ParamsSpec> Default for Ui<P> {
	fn default() -> Self {
		Self::new()
	}
}
