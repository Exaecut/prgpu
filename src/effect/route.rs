//! Router primitives: the [`Route`] trait the generated `Router` enum
//! implements, plus the per-instance route storage that bridges the route param
//! to an ergonomic `Router::current()`.
//!
//! The route is per **effect instance**. Because button handlers and label
//! exprs can't read a param directly, the adapter seeds [`CURRENT`](current_index)
//! at the start of each command (from the requested/effective route for this
//! instance, falling back to the persisted popup param), and a requested route
//! ([`request_index`]) is flushed back to that param after a handler runs.
//!
//! `PENDING`/`EFFECTIVE` are keyed by the effect instance id
//! ([`current_instance_id`]) so concurrent instances — and the GPU render thread
//! setting a route that the PF/UI thread reads — never clobber each other.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::effect::instance::current_instance_id;

/// Implemented by the `params!`-generated `Router` enum. Maps each route to a
/// stable `u32` index (declaration order) used for storage + group gating.
pub trait Route: Copy + 'static {
	const ALL: &'static [Self];
	const INITIAL: Self;
	fn to_index(self) -> u32;
	fn from_index(index: u32) -> Self;
	fn name(self) -> &'static str;
}

thread_local! {
	// Per-call active route, seeded on the processing thread for this instance.
	static CURRENT: Cell<u32> = const { Cell::new(0) };
}

// Requested route per instance: a GPU-render or background-task thread can ask
// for a route change; the main thread flushes it to the per-instance route param
// on the next UpdateParamsUi.
static PENDING: Mutex<BTreeMap<i32, u32>> = Mutex::new(BTreeMap::new());

// Authoritative within-session route per instance, set on every flush. A param
// write made outside PF_Cmd_USER_CHANGED_PARAM isn't guaranteed to commit to the
// project, so EFFECTIVE is the session source of truth that survives passes; the
// popup param is best-effort persistence across project reopen.
static EFFECTIVE: Mutex<BTreeMap<i32, u32>> = Mutex::new(BTreeMap::new());

/// Active route index for the instance currently being processed. Read by the
/// generated `Router::current()`, label exprs, and the route-visibility pass.
pub fn current_index() -> u32 {
	CURRENT.with(|c| c.get())
}

/// Adapter-only: seed the active route for this command from the route param.
pub fn set_current_index(index: u32) {
	CURRENT.with(|c| c.set(index));
}

/// Record a requested route (from `Router::set` / `ActionCtx::goto`, any thread)
/// for the current instance. The adapter flushes it to the route param on the
/// next UpdateParamsUi, then re-applies visibility.
pub fn request_index(index: u32) {
	PENDING.lock().unwrap().insert(current_instance_id(), index);
	crate::effect::labels::mark_dirty();
}

/// Adapter-only: take + clear this instance's pending route, if any.
pub fn take_pending_index() -> Option<u32> {
	PENDING.lock().unwrap().remove(&current_instance_id())
}

/// Whether a route change is awaiting flush for the current instance.
pub fn has_pending() -> bool {
	PENDING.lock().unwrap().contains_key(&current_instance_id())
}

/// Adapter-only: record the authoritative session route for this instance.
pub fn set_effective_index(index: u32) {
	EFFECTIVE.lock().unwrap().insert(current_instance_id(), index);
}

/// Adapter-only: this instance's authoritative session route, if any was
/// flushed. `None` ⇒ seed from the persisted popup store instead.
pub fn effective_index() -> Option<u32> {
	EFFECTIVE.lock().unwrap().get(&current_instance_id()).copied()
}
