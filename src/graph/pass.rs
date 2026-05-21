//! Internal pass representation.
//!
//! User-facing builders (`graph.add_pass`, `graph.add_mip_chain`) translate
//! into these `PassDecl` variants. Each variant carries a type-erased
//! dispatcher closure built from the typed `Kernel<P>` + the user's
//! per-frame params closure, so the executor can run every pass through a
//! uniform interface.

use crate::effect::FrameBinding;
use crate::graph::context::PassContext;
use crate::graph::resource::ResourceId;

/// Source / target binding the executor resolves per-pass.
///
/// `MainSource` / `Output` resolve through the active `InvocationBase`;
/// `ResourceMip` / `ResourceWhole` resolve through the executor's resource
/// table; `Inline` carries a pre-built `FrameBinding` (used by the executor
/// when promoting a snapshot or test fixture).
#[derive(Debug, Clone, Copy)]
pub enum Slot {
	MainSource,
	Output,
	ResourceMip(ResourceId, u32),
	ResourceWhole(ResourceId),
	Inline(FrameBinding),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipDirection {
	Down,
	Up,
}

/// Type-erased single-pass dispatcher: `(ctx, config) → Result<()>`.
pub type SingleDispatcher<F> = Box<dyn Fn(&PassContext<F>, &crate::types::Configuration) -> Result<(), &'static str> + Send + Sync + 'static>;

/// Type-erased mip-chain dispatcher: `(level, ctx, config) → Result<()>`.
pub type MipDispatcher<F> = Box<dyn Fn(u32, &PassContext<F>, &crate::types::Configuration) -> Result<(), &'static str> + Send + Sync + 'static>;

pub(crate) struct SinglePassDecl<F> {
	pub name: &'static str,
	pub source: Slot,
	pub input: Option<Slot>,
	pub target: Slot,
	pub dispatcher: SingleDispatcher<F>,
}

pub(crate) struct MipChainPassDecl<F> {
	pub name: &'static str,
	pub resource: ResourceId,
	pub direction: MipDirection,
	pub dispatcher: MipDispatcher<F>,
}

pub(crate) enum PassDecl<F> {
	Single(SinglePassDecl<F>),
	MipChain(MipChainPassDecl<F>),
}

impl<F> PassDecl<F> {
	pub fn name(&self) -> &'static str {
		match self {
			PassDecl::Single(p) => p.name,
			PassDecl::MipChain(p) => p.name,
		}
	}
}
