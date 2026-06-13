//! Pass-builder graph API (`Graph<P>`).
//!
//! Replaces `RenderGraph<F>` with a `ParamsSpec`-parameterised graph builder.
//! Pipelines declare resources and passes via method chaining; the executor
//! resolves slots and dispatches against a per-frame `Ctx<P>`.

use std::marker::PhantomData;

use crate::effect::Ctx;
use crate::graph::pass::{MipDirection, PyramidHandle, SingleDispatcher, MipDispatcher, EnabledPredicate, SinglePassDecl, MipChainPassDecl, PassDecl, Slot};
use crate::graph::resource::{MipPyramidDesc, ResourceId};
use crate::graph::source::SourcePolicy;
use crate::kernel::KernelParams;
use crate::kernel::{FromCtx, Kernel};
use crate::params::ParamsSpec;

#[derive(Clone)]
pub struct Derived<T: Send + Sync + 'static> {
	pub(crate) index: usize,
	_marker: PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> Copy for Derived<T> {}

pub(crate) struct ResourceDecl<P: ParamsSpec> {
	pub name: &'static str,
	pub desc_fn: Box<dyn Fn(&Ctx<P>) -> MipPyramidDesc + Send + Sync + 'static>,
}

pub(crate) struct DerivedDecl<P: ParamsSpec> {
	pub compute: Box<dyn Fn(&Ctx<P>) -> Box<dyn std::any::Any + Send + Sync> + Send + Sync + 'static>,
}

pub struct Graph<P: ParamsSpec> {
	pub(crate) source_policy: SourcePolicy,
	pub(crate) resources: Vec<ResourceDecl<P>>,
	pub(crate) passes: Vec<PassDecl<P>>,
	pub(crate) derived: Vec<DerivedDecl<P>>,
}

impl<P: ParamsSpec> Graph<P> {
	pub fn new() -> Self {
		Self {
			source_policy: SourcePolicy::default(),
			resources: Vec::new(),
			passes: Vec::new(),
			derived: Vec::new(),
		}
	}

	pub fn source_policy(&mut self, p: SourcePolicy) {
		self.source_policy = p;
	}

	pub fn derive<T, F>(&mut self, f: F) -> Derived<T>
	where
		T: Send + Sync + 'static,
		F: Fn(&Ctx<P>) -> T + Send + Sync + 'static,
	{
		let index = self.derived.len();
		self.derived.push(DerivedDecl {
			compute: Box::new(move |ctx| Box::new(f(ctx))),
		});
		Derived { index, _marker: PhantomData }
	}

	pub fn mip_pyramid<F>(&mut self, name: &'static str, desc_fn: F) -> PyramidHandle
	where
		F: Fn(&Ctx<P>) -> MipPyramidDesc + Send + Sync + 'static,
	{
		let id = ResourceId(self.resources.len() as u32);
		self.resources.push(ResourceDecl {
			name,
			desc_fn: Box::new(desc_fn),
		});
		PyramidHandle { id }
	}

	pub fn pass<K>(&mut self, kernel: Kernel<K>) -> PassBuilder<'_, P, K>
	where
		K: KernelParams + FromCtx<Spec = P>,
	{
		let default_params: Box<dyn Fn(&Ctx<P>) -> K + Send + Sync + 'static> =
			Box::new(|ctx| K::from_ctx(ctx));
		PassBuilder {
			graph: self,
			name: kernel.name(),
			kernel,
			source: Slot::Source,
			input: None,
			target: Slot::Output,
			params_fn: Some(default_params),
			enabled_when: None,
		}
	}

	pub fn pass_with<K, F>(&mut self, kernel: Kernel<K>, params_fn: F) -> PassBuilder<'_, P, K>
	where
		K: KernelParams,
		F: Fn(&Ctx<P>) -> K + Send + Sync + 'static,
	{
		PassBuilder {
			graph: self,
			name: kernel.name(),
			kernel,
			source: Slot::Source,
			input: None,
			target: Slot::Output,
			params_fn: Some(Box::new(params_fn)),
			enabled_when: None,
		}
	}

	pub fn mip_chain<K>(
		&mut self,
		pyramid: PyramidHandle,
		dir: MipDirection,
		kernel: Kernel<K>,
	) -> MipChainBuilder<'_, P, K>
	where
		K: KernelParams,
	{
		MipChainBuilder {
			graph: self,
			name: kernel.name(),
			pyramid: pyramid.id,
			direction: dir,
			kernel,
			params_fn: None,
			enabled_when: None,
		}
	}

	#[doc(hidden)]
	pub fn pass_count(&self) -> usize {
		self.passes.len()
	}

	#[doc(hidden)]
	pub fn resource_count(&self) -> usize {
		self.resources.len()
	}
}

pub struct PassBuilder<'g, P: ParamsSpec, K: KernelParams> {
	graph: &'g mut Graph<P>,
	name: &'static str,
	kernel: Kernel<K>,
	source: Slot,
	input: Option<Slot>,
	target: Slot,
	params_fn: Option<Box<dyn Fn(&Ctx<P>) -> K + Send + Sync + 'static>>,
	enabled_when: Option<Box<dyn Fn(&Ctx<P>) -> bool + Send + Sync + 'static>>,
}

impl<'g, P: ParamsSpec, K: KernelParams + Send + Sync + 'static> PassBuilder<'g, P, K> {
	pub fn reads(mut self, s: Slot) -> Self {
		self.source = s;
		self
	}

	pub fn reads_input(mut self, s: Slot) -> Self {
		self.input = Some(s);
		self
	}

	pub fn writes(mut self, s: Slot) -> Self {
		self.target = s;
		self
	}

	pub fn params<F>(mut self, f: F) -> Self
	where
		F: Fn(&Ctx<P>) -> K + Send + Sync + 'static,
	{
		self.params_fn = Some(Box::new(f));
		self
	}

	pub fn when<F>(mut self, f: F) -> Self
	where
		F: Fn(&Ctx<P>) -> bool + Send + Sync + 'static,
	{
		self.enabled_when = Some(Box::new(f));
		self
	}
}

impl<P: ParamsSpec, K: KernelParams + Send + Sync + 'static> Drop for PassBuilder<'_, P, K> {
	fn drop(&mut self) {
		let name = self.name;
		let kernel = self.kernel.clone();
		let source = self.source;
		let input = self.input;
		let target = self.target;
		let params_fn = self.params_fn.take().unwrap();
		let enabled_when = self.enabled_when.take();

		let dispatcher: SingleDispatcher<P> = Box::new(move |ctx, config| {
			let params = params_fn(ctx);
			match ctx.capabilities().backend() {
				crate::types::Backend::Cpu => {
					unsafe { kernel.dispatch_cpu_direct(config, params) };
					Ok(())
				}
				crate::types::Backend::Cuda | crate::types::Backend::Metal => unsafe { kernel.dispatch_gpu(config, params) },
			}
		});

		self.graph.passes.push(PassDecl::Single(SinglePassDecl {
			name,
			source,
			input,
			target,
			dispatcher,
			enabled_when,
		}));
	}
}

pub struct MipChainBuilder<'g, P: ParamsSpec, K: KernelParams> {
	graph: &'g mut Graph<P>,
	name: &'static str,
	pyramid: ResourceId,
	direction: MipDirection,
	kernel: Kernel<K>,
	params_fn: Option<Box<dyn Fn(u32, &Ctx<P>) -> K + Send + Sync + 'static>>,
	enabled_when: Option<Box<dyn Fn(&Ctx<P>) -> bool + Send + Sync + 'static>>,
}

impl<'g, P: ParamsSpec, K: KernelParams + Send + Sync + 'static> MipChainBuilder<'g, P, K> {
	pub fn params<F>(mut self, f: F) -> Self
	where
		F: Fn(u32, &Ctx<P>) -> K + Send + Sync + 'static,
	{
		self.params_fn = Some(Box::new(f));
		self
	}

	pub fn when<F>(mut self, f: F) -> Self
	where
		F: Fn(&Ctx<P>) -> bool + Send + Sync + 'static,
	{
		self.enabled_when = Some(Box::new(f));
		self
	}
}

impl<P: ParamsSpec, K: KernelParams + Send + Sync + 'static> Drop for MipChainBuilder<'_, P, K> {
	fn drop(&mut self) {
		let name = self.name;
		let pyramid = self.pyramid;
		let direction = self.direction;
		let kernel = self.kernel.clone();
		let params_fn = self.params_fn.take().unwrap();
		let enabled_when = self.enabled_when.take();

		let dispatcher: MipDispatcher<P> = Box::new(move |level, ctx, config| {
			let params = params_fn(level, ctx);
			match ctx.capabilities().backend() {
				crate::types::Backend::Cpu => {
					unsafe { kernel.dispatch_cpu_direct(config, params) };
					Ok(())
				}
				crate::types::Backend::Cuda | crate::types::Backend::Metal => unsafe { kernel.dispatch_gpu(config, params) },
			}
		});

		self.graph.passes.push(PassDecl::MipChain(MipChainPassDecl {
			name,
			resource: pyramid,
			direction,
			dispatcher,
			enabled_when,
		}));
	}
}
