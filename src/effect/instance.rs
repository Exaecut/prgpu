//! Per-call "current effect instance id", seeded by the adapters on both entry
//! points so per-instance state (route, effect state) resolves to the right
//! instance. On Premiere the id is `GetFilterInstanceID` on the PF side and the
//! `Effect_RuntimeInstanceID` property on the GPU side — the same id used to key
//! the `OpaqueEffectData` shared blob. 0 = unknown (After Effects / pre-seed).

use std::cell::Cell;

thread_local! {
	static CURRENT: Cell<i32> = const { Cell::new(0) };
}

/// Adapter-only: seed the active instance id for this command/render.
pub fn set_current_instance_id(id: i32) {
	CURRENT.with(|c| c.set(id));
}

/// The effect instance currently being processed on this thread. Keys the
/// per-instance state shared between the UI (PF) and render (GPU) entry points.
pub fn current_instance_id() -> i32 {
	CURRENT.with(|c| c.get())
}
