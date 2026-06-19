//! Thread-safe stash for label text produced by `Ui::set_label` closures.
//!
//! `apply_visibility` evaluates the closures (which need `&Ctx<P>`) and stores
//! the resulting `String` here keyed by the param's declaration index. The
//! `PF_Event_DRAW` handler reads it back to know what to paint via Drawbot.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use parking_lot::RwLock;

// Retained-mode "UI needs redraw" flag. Any state/param/label mutation sets it
// (from any thread); the adapter consumes it on the next host command and sets
// PF_OutFlag_REFRESH_UI so the ECW repaints. Not a timer — refresh is driven by
// mutation, delivered on the next host callback.
static DIRTY: AtomicBool = AtomicBool::new(false);

/// Mark the UI dirty (a mutation happened, from any thread). The adapter
/// consumes this on the next host command and sets PF_OutFlag_REFRESH_UI.
///
/// NOTE: we do NOT poke the host from here. Per the SDK (`background_task`
/// example), UI/register suites are main-thread only — calling them from a task
/// worker thread is undefined behaviour. Premiere offers no background-thread
/// refresh (its live-UI mechanisms, the idle hook and the RenderAsyncManager,
/// are After-Effects-only), so the redraw lands on the next host callback.
pub fn mark_dirty() {
	DIRTY.store(true, Ordering::Relaxed);
}

/// Adapter-only: take + clear the dirty flag.
pub fn take_dirty() -> bool {
	DIRTY.swap(false, Ordering::Relaxed)
}

static STASH: OnceLock<RwLock<HashMap<usize, String>>> = OnceLock::new();

fn map() -> &'static RwLock<HashMap<usize, String>> {
	STASH.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn set(index: usize, text: &str) {
	map().write().insert(index, text.to_string());
}

pub fn get(index: usize) -> Option<String> {
	map().read().get(&index).cloned()
}

/// Placeholder arbitrary-data payload for `#[label]` params.
///
/// On Premiere a custom-UI param must be ARBITRARY (or null), not a standard
/// type — and empirically only the arbitrary form receives `PF_Event_DRAW`
/// (`Custom_ECW_UI.cpp` uses `PF_ADD_ARBITRARY2` on its Premiere branch). The
/// label carries no real data; this zero-byte payload exists only so the host
/// will allocate the param and send draw events. The text is drawn from the
/// stash above, not from this value.
#[repr(C)]
#[derive(Default, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct LabelArb(pub u8);

impl after_effects::ArbitraryData<LabelArb> for LabelArb {
	fn interpolate(&self, _other: &LabelArb, _t: f64) -> LabelArb {
		*self
	}
}
