//! Graph-managed resources (mip pyramids, scratch images).
//!
//! Resource handles are typed via a `Kind` marker so the slot DSL can reject
//! `bloom.mip(0)` calls on a non-pyramid resource at compile time. The
//! executor allocates each declared resource via `cpu::buffer` /
//! `gpu::buffer` and stores the resulting [`crate::types::ImageBuffer`]
//! against the handle's id.

use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub(crate) u32);

/// Strongly-typed handle into the graph's resource table. The `Kind` is a
/// zero-sized marker (`MipPyramid`, ...) so the slot DSL can guarantee mip
/// access only happens on real pyramids.
pub struct ResourceHandle<Kind> {
	pub(crate) id: ResourceId,
	_kind: PhantomData<fn() -> Kind>,
}

impl<Kind> Clone for ResourceHandle<Kind> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<Kind> Copy for ResourceHandle<Kind> {}

impl<Kind> ResourceHandle<Kind> {
	pub(crate) const fn new(id: ResourceId) -> Self {
		Self { id, _kind: PhantomData }
	}

	pub const fn id(&self) -> ResourceId {
		self.id
	}
}

/// Marker for an N-level mip pyramid resource.
pub struct MipPyramid;

impl ResourceHandle<MipPyramid> {
	pub fn mip(self, lod: u32) -> crate::graph::pass::Slot {
		crate::graph::pass::Slot::ResourceMip(self.id, lod)
	}

	pub fn whole(self) -> crate::graph::pass::Slot {
		crate::graph::pass::Slot::ResourceWhole(self.id)
	}
}

/// How long the executor keeps a resource alive across renders. The CPU
/// buffer pool currently only honours `Device` (= "keyed by tag, kept warm
/// across calls"); other variants fall back to that for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceLifetime {
	Frame,
	EffectInstance,
	Device,
	Global,
}

impl Default for ResourceLifetime {
	fn default() -> Self {
		ResourceLifetime::Device
	}
}

#[derive(Debug, Clone, Copy)]
pub struct MipPyramidDesc {
	pub base_width: u32,
	pub base_height: u32,
	pub levels: u32,
	pub tag: u32,
	pub lifetime: ResourceLifetime,
	pub populate_from_source: bool,
}

impl MipPyramidDesc {
	pub const fn new(base_width: u32, base_height: u32) -> Self {
		Self {
			base_width,
			base_height,
			levels: 1,
			tag: 0,
			lifetime: ResourceLifetime::Device,
			populate_from_source: false,
		}
	}

	pub const fn levels(mut self, levels: u32) -> Self {
		self.levels = levels;
		self
	}

	pub const fn tag(mut self, tag: u32) -> Self {
		self.tag = tag;
		self
	}

	pub const fn lifetime(mut self, lifetime: ResourceLifetime) -> Self {
		self.lifetime = lifetime;
		self
	}

	pub const fn populate_from_source(mut self, populate: bool) -> Self {
		self.populate_from_source = populate;
		self
	}
}
