//! Imperative graph-builder API.
//!
//! `RenderGraph<F>` is built once per effect-instance lifetime by an
//! `Effect::pipeline` callback (Phase 11). The current Phase 4 surface
//! exposes three declarations:
//!
//! - [`RenderGraph::declare_mip_pyramid`] — register a sized N-level mip
//!   pyramid sized by a per-frame closure.
//! - [`RenderGraph::add_pass`] — register a single source/(input)/target
//!   kernel pass.
//! - [`RenderGraph::add_mip_chain`] — register a per-level sweep across a
//!   mip pyramid (down or up direction).
//!
//! Validation runs on first execution; declaration-time errors surface as
//! [`GraphError`] when the executor walks the graph.

use crate::graph::context::{MipPyramidCtx, PassContext};
use crate::graph::pass::{MipChainPassDecl, MipDirection, MipDispatcher, PassDecl, SingleDispatcher, SinglePassDecl, Slot};
use crate::graph::resource::{MipPyramid, MipPyramidDesc, ResourceHandle, ResourceId};
use crate::graph::source::SourcePolicy;
use crate::kernel::Kernel;
use crate::kernel::KernelParams;
use crate::types::ConfigBuildError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
	MissingTarget(&'static str),
	UnknownResource(&'static str),
	BadMipLevel { pass: &'static str, level: u32, max: u32 },
	ConfigBuild { pass: &'static str, kind: ConfigBuildError },
	KernelDispatch { pass: &'static str, message: &'static str },
	ResourceAllocFailed { name: &'static str },
}

pub(crate) struct ResourceDecl<F> {
	pub name: &'static str,
	pub desc_fn: Box<dyn Fn(&MipPyramidCtx<F>) -> MipPyramidDesc + Send + Sync + 'static>,
}

pub struct RenderGraph<F: Copy + Send + Sync + 'static> {
	pub(crate) source_policy: SourcePolicy,
	pub(crate) resources: Vec<ResourceDecl<F>>,
	pub(crate) passes: Vec<PassDecl<F>>,
}

impl<F: Copy + Send + Sync + 'static> RenderGraph<F> {
	#[doc(hidden)]
	pub fn pass_count(&self) -> usize {
		self.passes.len()
	}

	#[doc(hidden)]
	pub fn resource_count(&self) -> usize {
		self.resources.len()
	}
}

impl<F: Copy + Send + Sync + 'static> Default for RenderGraph<F> {
	fn default() -> Self {
		Self::new()
	}
}

impl<F: Copy + Send + Sync + 'static> RenderGraph<F> {
	pub fn new() -> Self {
		Self {
			source_policy: SourcePolicy::Direct,
			resources: Vec::new(),
			passes: Vec::new(),
		}
	}

	pub fn set_source_policy(&mut self, policy: SourcePolicy) {
		self.source_policy = policy;
	}

	pub fn declare_mip_pyramid<R>(&mut self, name: &'static str, desc_fn: R) -> ResourceHandle<MipPyramid>
	where
		R: Fn(&MipPyramidCtx<F>) -> MipPyramidDesc + Send + Sync + 'static,
	{
		let id = ResourceId(self.resources.len() as u32);
		self.resources.push(ResourceDecl {
			name,
			desc_fn: Box::new(desc_fn),
		});
		ResourceHandle::<MipPyramid>::new(id)
	}

	/// Register a single-pass kernel invocation.
	///
	/// The `params_fn` closure runs per frame, pulling values out of the
	/// effect's `FrameData` to build the kernel's per-pass constant buffer.
	pub fn add_pass<P, A>(&mut self, name: &'static str, kernel: Kernel<P>, source: Slot, target: Slot, params_fn: A)
	where
		P: KernelParams + Send + Sync,
		A: Fn(&PassContext<F>) -> P + Send + Sync + 'static,
	{
		self.add_pass_inner(name, kernel, source, None, target, params_fn, None);
	}

	/// Register a single-pass kernel invocation with a conditional predicate.
	pub fn add_pass_conditional<P, A, E>(&mut self, name: &'static str, kernel: Kernel<P>, source: Slot, target: Slot, params_fn: A, enabled_fn: E)
	where
		P: KernelParams + Send + Sync,
		A: Fn(&PassContext<F>) -> P + Send + Sync + 'static,
		E: Fn(&PassContext<F>) -> bool + Send + Sync + 'static,
	{
		self.add_pass_inner(name, kernel, source, None, target, params_fn, Some(Box::new(enabled_fn)));
	}

	/// Register a single-pass invocation with an explicit secondary input
	/// (slot 1 / `incoming`). Used when a kernel reads two distinct sources
	/// (e.g. composite reads source + bloom-pyramid).
	pub fn add_pass_with_input<P, A>(&mut self, name: &'static str, kernel: Kernel<P>, source: Slot, input: Slot, target: Slot, params_fn: A)
	where
		P: KernelParams + Send + Sync,
		A: Fn(&PassContext<F>) -> P + Send + Sync + 'static,
	{
		self.add_pass_inner(name, kernel, source, Some(input), target, params_fn, None);
	}

	/// Register a single-pass invocation with an explicit secondary input and a conditional predicate.
	pub fn add_pass_with_input_conditional<P, A, E>(&mut self, name: &'static str, kernel: Kernel<P>, source: Slot, input: Slot, target: Slot, params_fn: A, enabled_fn: E)
	where
		P: KernelParams + Send + Sync,
		A: Fn(&PassContext<F>) -> P + Send + Sync + 'static,
		E: Fn(&PassContext<F>) -> bool + Send + Sync + 'static,
	{
		self.add_pass_inner(name, kernel, source, Some(input), target, params_fn, Some(Box::new(enabled_fn)));
	}

	fn add_pass_inner<P, A>(&mut self, name: &'static str, kernel: Kernel<P>, source: Slot, input: Option<Slot>, target: Slot, params_fn: A, enabled_when: Option<crate::graph::pass::EnabledPredicate<F>>)
	where
		P: KernelParams + Send + Sync,
		A: Fn(&PassContext<F>) -> P + Send + Sync + 'static,
	{
		let dispatcher: SingleDispatcher<F> = Box::new(move |ctx, config| {
			let params = params_fn(ctx);
			match ctx.capabilities().backend() {
				crate::types::Backend::Cpu => {
					unsafe { kernel.dispatch_cpu_direct(config, params) };
					Ok(())
				}
				crate::types::Backend::Cuda | crate::types::Backend::Metal | crate::types::Backend::OpenCL => unsafe { kernel.dispatch_gpu(config, params) },
			}
		});

		self.passes.push(PassDecl::Single(SinglePassDecl {
			name,
			source,
			input,
			target,
			dispatcher,
			enabled_when,
		}));
	}

	/// Register an N-level sweep across a mip pyramid. `direction = Down`
	/// runs `lod 0 → 1 → ... → N-1`; `direction = Up` runs the reverse.
	pub fn add_mip_chain<P, A>(&mut self, name: &'static str, resource: ResourceHandle<MipPyramid>, direction: MipDirection, kernel: Kernel<P>, params_fn: A)
	where
		P: KernelParams + Send + Sync,
		A: Fn(u32, &PassContext<F>) -> P + Send + Sync + 'static,
	{
		self.add_mip_chain_inner(name, resource, direction, kernel, params_fn, None);
	}

	/// Register a conditional N-level sweep across a mip pyramid.
	pub fn add_mip_chain_conditional<P, A, E>(&mut self, name: &'static str, resource: ResourceHandle<MipPyramid>, direction: MipDirection, kernel: Kernel<P>, params_fn: A, enabled_fn: E)
	where
		P: KernelParams + Send + Sync,
		A: Fn(u32, &PassContext<F>) -> P + Send + Sync + 'static,
		E: Fn(&PassContext<F>) -> bool + Send + Sync + 'static,
	{
		self.add_mip_chain_inner(name, resource, direction, kernel, params_fn, Some(Box::new(enabled_fn)));
	}

	fn add_mip_chain_inner<P, A>(&mut self, name: &'static str, resource: ResourceHandle<MipPyramid>, direction: MipDirection, kernel: Kernel<P>, params_fn: A, enabled_when: Option<crate::graph::pass::EnabledPredicate<F>>)
	where
		P: KernelParams + Send + Sync,
		A: Fn(u32, &PassContext<F>) -> P + Send + Sync + 'static,
	{
		let dispatcher: MipDispatcher<F> = Box::new(move |level, ctx, config| {
			let params = params_fn(level, ctx);
			match ctx.capabilities().backend() {
				crate::types::Backend::Cpu => {
					unsafe { kernel.dispatch_cpu_direct(config, params) };
					Ok(())
				}
				crate::types::Backend::Cuda | crate::types::Backend::Metal | crate::types::Backend::OpenCL => unsafe { kernel.dispatch_gpu(config, params) },
			}
		});

		self.passes.push(PassDecl::MipChain(MipChainPassDecl {
			name,
			resource: resource.id(),
			direction,
			dispatcher,
			enabled_when,
		}));
	}
}
