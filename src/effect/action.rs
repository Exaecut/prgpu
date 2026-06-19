//! [`ActionCtx`] — the capability handle passed to button `on_action` handlers.
//! Decouples UI logic (route switching, background work) from UI declaration.

use std::marker::PhantomData;

use crate::effect::route::{self, Route};
use crate::effect::tasks::{self, BackgroundTask, TaskHandle, TaskId};
use crate::params::ParamsSpec;

/// Handed to a button's `on_action` handler. Exposes route navigation and the
/// background-task surface. Construction is adapter-internal.
pub struct ActionCtx<P: ParamsSpec> {
	_p: PhantomData<P>,
}

impl<P: ParamsSpec> ActionCtx<P> {
	#[doc(hidden)]
	pub fn __new() -> Self {
		Self { _p: PhantomData }
	}

	/// Navigate to any route. The adapter flushes the request to the
	/// per-instance route param and re-applies visibility after the handler
	/// returns.
	pub fn goto<R: Route>(&mut self, route: R) {
		route::request_index(route.to_index());
	}

	/// Start a background task, tagged for later group lookup/cancellation.
	pub fn spawn<T: BackgroundTask>(&mut self, task: T, tags: &[&'static str]) -> TaskHandle {
		tasks::spawn(task, tags)
	}

	/// Cancel a specific task by id.
	pub fn cancel(&mut self, id: TaskId) {
		tasks::cancel(id);
	}

	/// Cancel every task carrying `tag`.
	pub fn cancel_tag(&mut self, tag: &'static str) {
		tasks::cancel_tag(tag);
	}
}
