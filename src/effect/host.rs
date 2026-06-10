//! Static host + backend capability map.
//!
//! Effect code expresses contracts (`if !ctx.supports(Capability::FrameExpansion) { ... }`)
//! instead of `is_premiere()` / `is_after_effects()` checks scattered across the
//! codebase. Capabilities are derived from `(Host, Backend)` via fixed rules
//! the adapter layer applies on every render call.

use crate::types::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
	AfterEffects,
	Premiere,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderKind {
	AeLegacyRender,
	AeSmartRenderCpu,
	AeSmartRenderGpu,
	PremiereGpuEffect,
	PremiereGpuTransition,
	TestCpu,
	TestGpu,
}

impl RenderKind {
	pub const fn host(self) -> Host {
		match self {
			RenderKind::AeLegacyRender | RenderKind::AeSmartRenderCpu | RenderKind::AeSmartRenderGpu => Host::AfterEffects,
			RenderKind::PremiereGpuEffect | RenderKind::PremiereGpuTransition => Host::Premiere,
			RenderKind::TestCpu | RenderKind::TestGpu => Host::AfterEffects,
		}
	}

	pub const fn is_test(self) -> bool {
		matches!(self, RenderKind::TestCpu | RenderKind::TestGpu)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
	/// Effect can request a larger output rect than the input layer (AE only).
	FrameExpansion,
	/// Host respects per-parameter UI visibility flags during interactive editing.
	DynamicParamVisibility,
	/// Host may hand the same PPix as both source and output; passes that touch
	/// source after writing the output need a private snapshot.
	SourceOutputMayAlias,
	/// Native Premiere GPU filter dispatch (`xGPUFilterEntry`) is in play.
	NativePremiereGpuFilter,
	/// Effect is being driven through After Effects' `SmartRenderGpu` selector.
	SmartRenderGpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCapabilities {
	host: Host,
	backend: Backend,
}

impl HostCapabilities {
	pub const fn new(host: Host, backend: Backend) -> Self {
		Self { host, backend }
	}

	pub const fn host(&self) -> Host {
		self.host
	}

	pub const fn backend(&self) -> Backend {
		self.backend
	}

	pub const fn supports(&self, capability: Capability) -> bool {
		match capability {
			// AE: SmartFX `max_result_rect`. Premiere GPU filter: the outFrame
			// is the full sequence canvas, so out-of-clip pixels exist there
			// too; only Premiere CPU stays
			// clip-locked.
			Capability::FrameExpansion => matches!(self.host, Host::AfterEffects) || (matches!(self.host, Host::Premiere) && !matches!(self.backend, Backend::Cpu)),
			Capability::DynamicParamVisibility => true,
			Capability::SourceOutputMayAlias => matches!(self.host, Host::Premiere) && !matches!(self.backend, Backend::Cpu),
			Capability::NativePremiereGpuFilter => matches!(self.host, Host::Premiere) && !matches!(self.backend, Backend::Cpu),
			Capability::SmartRenderGpu => matches!(self.host, Host::AfterEffects) && !matches!(self.backend, Backend::Cpu),
		}
	}
}
