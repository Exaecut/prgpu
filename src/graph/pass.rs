//! Internal pass representation.
//!
//! User-facing builders (`Graph::pass`, `Graph::mip_chain`) translate into
//! these `PassDecl` variants. The executor resolves slots and dispatches
//! through a uniform interface parameterised on `P: ParamsSpec`.

use crate::effect::Ctx;
use crate::params::ParamsSpec;

/// Source / target binding the executor resolves per-pass.
#[derive(Debug, Clone, Copy)]
pub enum Slot {
	Source,
	Output,
	Mip(PyramidHandle, u32),
	#[doc(hidden)]
	Inline(crate::effect::FrameBinding),
}

/// Handle returned by `Graph::mip_pyramid`. Wraps an internal resource id.
#[derive(Debug, Clone, Copy)]
pub struct PyramidHandle {
	pub(crate) id: crate::graph::resource::ResourceId,
}

impl PyramidHandle {
	pub fn mip(self, level: u32) -> Slot {
		Slot::Mip(self, level)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipDirection {
	Down,
	Up,
}

/// Type-erased single-pass dispatcher.
pub type SingleDispatcher<P> = Box<dyn Fn(&Ctx<P>, &crate::types::Configuration) -> Result<(), &'static str> + Send + Sync + 'static>;

/// Type-erased mip-chain dispatcher.
pub type MipDispatcher<P> = Box<dyn Fn(u32, &Ctx<P>, &crate::types::Configuration) -> Result<(), &'static str> + Send + Sync + 'static>;

/// Optional pass predicate.
pub type EnabledPredicate<P> = Box<dyn Fn(&Ctx<P>) -> bool + Send + Sync + 'static>;

pub(crate) struct SinglePassDecl<P: ParamsSpec> {
	pub name: &'static str,
	pub source: Slot,
	pub input: Option<Slot>,
	pub target: Slot,
	pub dispatcher: SingleDispatcher<P>,
	pub enabled_when: Option<EnabledPredicate<P>>,
}

pub(crate) struct MipChainPassDecl<P: ParamsSpec> {
	pub name: &'static str,
	pub resource: crate::graph::resource::ResourceId,
	pub direction: MipDirection,
	pub dispatcher: MipDispatcher<P>,
	pub enabled_when: Option<EnabledPredicate<P>>,
}

pub(crate) enum PassDecl<P: ParamsSpec> {
	Single(SinglePassDecl<P>),
	MipChain(MipChainPassDecl<P>),
}

impl<P: ParamsSpec> PassDecl<P> {
	pub fn name(&self) -> &'static str {
		match self {
			PassDecl::Single(p) => p.name,
			PassDecl::MipChain(p) => p.name,
		}
	}
}
