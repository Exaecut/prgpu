//! Per-pass execution context.
//!
//! Closures the user supplies (param computation, mip-pyramid sizing,
//! `enabled_when` predicates) all receive a `*Ctx<F>` view that exposes
//! frame data plus host metadata in one place.

use crate::effect::{HostCapabilities, InvocationBase};

/// Context handed to per-pass param closures.
pub struct PassContext<'a, F> {
	frame_data: &'a F,
	base: &'a InvocationBase,
}

impl<'a, F> PassContext<'a, F> {
	pub(crate) fn new(frame_data: &'a F, base: &'a InvocationBase) -> Self {
		Self { frame_data, base }
	}

	#[inline]
	pub fn frame_data(&self) -> &F {
		self.frame_data
	}

	#[inline]
	pub fn capabilities(&self) -> HostCapabilities {
		self.base.capabilities()
	}

	#[inline]
	pub fn output_width(&self) -> u32 {
		self.base.output.width
	}

	#[inline]
	pub fn output_height(&self) -> u32 {
		self.base.output.height
	}

	#[inline]
	pub fn frame_index(&self) -> u32 {
		self.base.render_generation as u32
	}
}

/// Context handed to mip-pyramid descriptor closures (`graph.declare_mip_pyramid`).
pub struct MipPyramidCtx<'a, F> {
	frame_data: &'a F,
	base: &'a InvocationBase,
}

impl<'a, F> MipPyramidCtx<'a, F> {
	pub(crate) fn new(frame_data: &'a F, base: &'a InvocationBase) -> Self {
		Self { frame_data, base }
	}

	#[inline]
	pub fn frame_data(&self) -> &F {
		self.frame_data
	}

	#[inline]
	pub fn output_width(&self) -> u32 {
		self.base.output.width
	}

	#[inline]
	pub fn output_height(&self) -> u32 {
		self.base.output.height
	}

	#[inline]
	pub fn bytes_per_pixel(&self) -> u32 {
		self.base.bytes_per_pixel
	}
}
