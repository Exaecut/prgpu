//! Router primitives: the [`Route`] trait the generated `Router` enum
//! implements, plus the per-call thread-locals that bridge the per-instance
//! route param to an ergonomic `Router::current()`.
//!
//! The route is stored **per effect instance** in a hidden popup param
//! (project-persisted). Because button handlers and label exprs can't read a
//! param directly, the adapter seeds [`CURRENT`](current_index) from that param
//! at the start of each command, and flushes a requested route
//! ([`request_index`]) back to the param after a handler runs.

use std::cell::Cell;
use std::sync::atomic::{AtomicI64, Ordering};

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
	// Per-instance active route, seeded on the main thread from the route param.
	static CURRENT: Cell<u32> = const { Cell::new(0) };
}

// Requested route. GLOBAL (not thread-local) so a background-task worker thread
// can request a route change (e.g. "job done → Complete"); the main thread
// flushes it to the per-instance route param on the next UpdateParamsUi.
// -1 = no request. (Process-global ⇒ single-instance-accurate; with multiple
// instances the request applies to whichever next flushes — acceptable for the
// demo, revisit if per-instance background navigation is needed.)
static PENDING: AtomicI64 = AtomicI64::new(-1);

// Authoritative within-session route, set on every flush. The route store is a
// hidden popup param (Model B persistence), but a param-value write made
// outside PF_Cmd_USER_CHANGED_PARAM is not guaranteed to commit to the project
// — so a background-initiated change (job done → Complete) could be re-read as
// the old value on the next pass and the route would revert. EFFECTIVE is the
// session source of truth that survives passes; the popup is best-effort
// persistence across project reopen. -1 = unset ⇒ fall back to the popup.
// Process-global, same single-instance caveat as PENDING.
// TODO(multi-instance): move route storage into per-instance sequence_data.
static EFFECTIVE: AtomicI64 = AtomicI64::new(-1);

/// Active route index for the instance currently being processed. Read by the
/// generated `Router::current()`, label exprs, and the route-visibility pass.
pub fn current_index() -> u32 {
	CURRENT.with(|c| c.get())
}

/// Adapter-only: seed the active route for this command from the route param.
pub fn set_current_index(index: u32) {
	CURRENT.with(|c| c.set(index));
}

/// Record a requested route (from `Router::set` / `ActionCtx::goto`, any
/// thread). The adapter flushes it to the per-instance route param on the next
/// UpdateParamsUi, then re-applies visibility.
pub fn request_index(index: u32) {
	PENDING.store(index as i64, Ordering::SeqCst);
	crate::effect::labels::mark_dirty();
}

/// Adapter-only: take + clear the pending route, if any.
pub fn take_pending_index() -> Option<u32> {
	let v = PENDING.swap(-1, Ordering::SeqCst);
	(v >= 0).then_some(v as u32)
}

/// Whether a route change is awaiting flush (set from any thread, incl. a
/// background-task worker). The adapter re-applies visibility when true.
pub fn has_pending() -> bool {
	PENDING.load(Ordering::SeqCst) >= 0
}

/// Adapter-only: record the authoritative session route (called on flush). Lets
/// the route survive passes even when the hidden popup store doesn't persist a
/// background-initiated value change.
pub fn set_effective_index(index: u32) {
	EFFECTIVE.store(index as i64, Ordering::SeqCst);
}

/// Adapter-only: the authoritative session route, if any was flushed. `None`
/// ⇒ seed from the persisted popup store instead.
pub fn effective_index() -> Option<u32> {
	let v = EFFECTIVE.load(Ordering::SeqCst);
	(v >= 0).then_some(v as u32)
}
