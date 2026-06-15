//! The one read context. `Ctx::get(Marker)` is the typed, host-agnostic
//! parameter read used by `Effect` v2; it resolves a zero-sized [`Param`]
//! marker against the per-frame [`Snapshot`] with no locks or map lookups.
//!
//! Introduced here; the adapters keep building `FrameDataContext` until phase 5
//! swaps the trait/graph over and this replaces it.

use crate::effect::host::{Capability, HostCapabilities};
use crate::effect::invocation::MAX_AUX_LAYERS;
use crate::params::{FromParamValue, Param, ParamsSpec, Snapshot};
use crate::types::Backend;

#[derive(Clone, Copy, Debug, Default)]
pub struct Geometry {
	pub layer_w: u32,
	pub layer_h: u32,
	pub output_w: u32,
	pub output_h: u32,
	pub ext_x: i32,
	pub ext_y: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
	pub frame_index: u32,
	pub time_seconds: f32,
	pub progress: f32,
}

pub struct Ctx<'a, P: ParamsSpec> {
	snapshot: &'a P::Snapshot,
	geom: Geometry,
	time: Timing,
	caps: HostCapabilities,
	debug_view: bool,
	layers_present: [bool; MAX_AUX_LAYERS],
}

impl<'a, P: ParamsSpec> Ctx<'a, P> {
	pub fn new(snapshot: &'a P::Snapshot, geom: Geometry, time: Timing, caps: HostCapabilities, debug_view: bool) -> Self {
		Self {
			snapshot,
			geom,
			time,
			caps,
			debug_view,
			layers_present: [false; MAX_AUX_LAYERS],
		}
	}

	/// Reflect the adapter's secondary-input checkout into the read context so
	/// pipeline closures (`.when` / `.params`) can branch on whether an AE
	/// layer param was actually delivered. Set from
	/// [`crate::effect::InvocationBase::layer_presence`] on the render paths;
	/// defaults to all-absent (Premiere, expansion/visibility contexts).
	#[inline]
	pub fn set_layers_present(&mut self, layers_present: [bool; MAX_AUX_LAYERS]) {
		self.layers_present = layers_present;
	}

	/// Whether the secondary input at `index` (a `<Marker>::LAYER_INDEX`) was
	/// delivered by the host this frame.
	#[inline]
	pub fn layer_present(&self, index: u32) -> bool {
		self.layers_present.get(index as usize).copied().unwrap_or(false)
	}

	/// The read API. Fully typed via the marker: `ctx.get(Strength) -> f32`,
	/// `ctx.get(Tint) -> Color`, `ctx.get(QualityMode) -> Quality`. Compiles to
	/// an array index plus enum match.
	#[inline]
	pub fn get<M: Param<Spec = P>>(&self, _marker: M) -> M::Value {
		M::Value::from_value(self.snapshot.value(M::ID))
	}

	#[inline]
	pub fn supports(&self, capability: Capability) -> bool {
		self.caps.supports(capability)
	}

	#[inline]
	pub fn capabilities(&self) -> HostCapabilities {
		self.caps
	}

	#[inline]
	pub fn backend(&self) -> Backend {
		self.caps.backend()
	}

	#[inline]
	pub fn layer_size(&self) -> (u32, u32) {
		(self.geom.layer_w, self.geom.layer_h)
	}

	#[inline]
	pub fn output_size(&self) -> (u32, u32) {
		(self.geom.output_w, self.geom.output_h)
	}

	#[inline]
	pub fn ext(&self) -> (i32, i32) {
		(self.geom.ext_x, self.geom.ext_y)
	}

	#[inline]
	pub fn frame_index(&self) -> u32 {
		self.time.frame_index
	}

	#[inline]
	pub fn time_seconds(&self) -> f32 {
		self.time.time_seconds
	}

	#[inline]
	pub fn progress(&self) -> f32 {
		self.time.progress
	}

	#[inline]
	pub fn debug_view(&self) -> bool {
		self.debug_view
	}
}
