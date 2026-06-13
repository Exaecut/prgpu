//! Visibility + action plumbing the AE adapter applies on `UpdateParamsUi`.
//!
//! `ParamApi` lets effect authors declare per-parameter visibility predicates
//! and click-action callbacks once, instead of hand-writing the AE PF flag
//! flips and AEGP dynamic-stream toggles inside `handle_command`. The adapter
//! evaluates the registered predicates on every UI tick and pushes the
//! resulting `INVISIBLE` / `Hidden` flags through both surfaces.

use std::fmt::Debug;
use std::hash::Hash;

use after_effects::{InData, OutData, Parameters};

use crate::effect::HostCapabilities;
use crate::params::SetupParams;

/// Click-action callback signature. The `ActionContext` carries adapter-managed
/// side effects; more can be added as the API grows.
pub type ActionCallback = Box<dyn Fn(&mut ActionContext) -> Result<(), &'static str> + Send + Sync + 'static>;

/// Side-effects an action callback can request from the adapter.
/// Currently empty; reserved for future adapter-managed side effects.
pub struct ActionContext;

impl ActionContext {
	pub(crate) fn new() -> Self {
		Self
	}
}

pub(crate) struct VisibilityRule<P>
where
	P: Eq + Hash + Copy + Debug + 'static,
{
	pub param: P,
	pub predicate: Box<dyn Fn(&Parameters<P>, HostCapabilities) -> bool + Send + Sync + 'static>,
}

pub(crate) struct ActionRule<P> {
	pub param: P,
	pub callback: ActionCallback,
}

/// Effect-side parameter API: setup + visibility + click actions.
///
/// `Effect::params(p: &mut ParamApi<P>)` calls `p.raw_setup_mut()` to bind
/// the underlying `Parameters<P>` for `Params::setup`, then declares
/// visibility rules through `p.visibility(|v| { ... })` and click actions
/// through `p.actions(|a| { ... })`.
pub struct ParamApi<'a, P>
where
	P: SetupParams + Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	params: &'a mut Parameters<'a, P>,
	in_data: InData,
	out_data: OutData,
	visibility: Vec<VisibilityRule<P>>,
	actions: Vec<ActionRule<P>>,
}

impl<'a, P> ParamApi<'a, P>
where
	P: SetupParams + Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	pub fn new(params: &'a mut Parameters<'a, P>, in_data: InData, out_data: OutData) -> Self {
		Self {
			params,
			in_data,
			out_data,
			visibility: Vec::new(),
			actions: Vec::new(),
		}
	}

	/// Borrow the raw `Parameters<P>` for `Params::setup`. The adapter
	/// passes the same value back through this method during the AE
	/// `ParamSetup` selector.
	pub fn raw_setup_mut(&mut self) -> &mut Parameters<'a, P> {
		self.params
	}

	pub fn in_data(&self) -> &InData {
		&self.in_data
	}

	pub fn out_data(&mut self) -> &mut OutData {
		&mut self.out_data
	}

	/// Declare per-parameter visibility predicates. The adapter re-evaluates
	/// each predicate on `UpdateParamsUi` and toggles AE PF
	/// `ParamUIFlags::INVISIBLE` + AEGP `DynamicStreamFlags::Hidden`.
	pub fn visibility<F>(&mut self, f: F)
	where
		F: FnOnce(&mut VisibilityBuilder<P>),
	{
		let mut builder = VisibilityBuilder { rules: Vec::new() };
		f(&mut builder);
		self.visibility.extend(builder.rules);
	}

	/// Declare click handlers for button parameters. The adapter wires the
	/// callback into `Cmd_UserChangedParam`.
	pub fn actions<F>(&mut self, f: F)
	where
		F: FnOnce(&mut ActionBuilder<P>),
	{
		let mut builder = ActionBuilder { rules: Vec::new() };
		f(&mut builder);
		self.actions.extend(builder.rules);
	}

	pub(crate) fn into_rules(self) -> (Vec<VisibilityRule<P>>, Vec<ActionRule<P>>) {
		(self.visibility, self.actions)
	}
}

pub struct VisibilityBuilder<P>
where
	P: Eq + Hash + Copy + Debug + 'static,
{
	rules: Vec<VisibilityRule<P>>,
}

impl<P> VisibilityBuilder<P>
where
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	pub fn show<F>(&mut self, param: P, predicate: F) -> &mut Self
	where
		F: Fn(&Parameters<P>, HostCapabilities) -> bool + Send + Sync + 'static,
	{
		self.rules.push(VisibilityRule {
			param,
			predicate: Box::new(predicate),
		});
		self
	}

	pub fn show_all<I, F>(&mut self, params: I, predicate: F) -> &mut Self
	where
		I: IntoIterator<Item = P>,
		F: Fn(&Parameters<P>, HostCapabilities) -> bool + Clone + Send + Sync + 'static,
	{
		for p in params {
			let pred = predicate.clone();
			self.rules.push(VisibilityRule {
				param: p,
				predicate: Box::new(pred),
			});
		}
		self
	}
}

pub struct ActionBuilder<P> {
	rules: Vec<ActionRule<P>>,
}

impl<P> ActionBuilder<P>
where
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	pub fn on_click<F>(&mut self, param: P, callback: F) -> &mut Self
	where
		F: Fn(&mut ActionContext) -> Result<(), &'static str> + Send + Sync + 'static,
	{
		self.rules.push(ActionRule { param, callback: Box::new(callback) });
		self
	}
}
