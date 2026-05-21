//! Host-agnostic per-frame parameter view.
//!
//! `Effect::frame_data` and `Effect::expansion` both receive a context
//! that abstracts the AE PF / Premiere GPU parameter retrieval, popup
//! 0-base normalisation, time, and dimensions. The user calls
//! `ctx.float(Params::Radius)` and the right host extractor runs underneath.

use std::fmt::Debug;
use std::hash::Hash;

use after_effects::sys::PF_Pixel;
use after_effects::Parameters;
use premiere as pr;

use crate::effect::host::{Host, HostCapabilities, RenderKind};
use crate::params::{get_param, CpuParams, SetupParams};
use crate::types::{Backend, Pixel};

/// Per-host backend the context dispatches through. Carries the lifetime'd
/// host objects (CPU `Parameters<P>` or Premiere `GpuFilterData` +
/// `RenderParams`) so the public methods can extract values without
/// allocating.
pub(crate) enum HostBackend<'a, P>
where
	P: SetupParams,
{
	Cpu {
		params: &'a Parameters<'a, P>,
		is_premiere: bool,
	},
	Gpu {
		filter: &'a pr::GpuFilterData,
		render_params: &'a pr::RenderParams,
	},
}

/// Context handed to `Effect::frame_data` (and `Effect::expansion` via the
/// thinner [`ExpansionContext`] wrapper). Exposes parameter extractors,
/// dimensions, and timing in a single host-agnostic surface.
pub struct FrameDataContext<'a, P>
where
	P: SetupParams,
{
	pub(crate) host: Host,
	pub(crate) backend: Backend,
	pub(crate) render_kind: RenderKind,
	pub(crate) inner: HostBackend<'a, P>,
	pub(crate) layer_width: u32,
	pub(crate) layer_height: u32,
	pub(crate) output_width: u32,
	pub(crate) output_height: u32,
	pub(crate) frame_index: u32,
	pub(crate) time_seconds: f32,
	pub(crate) progress: f32,
}

impl<'a, P> FrameDataContext<'a, P>
where
	P: SetupParams + Eq + Hash + Copy + Debug + 'static,
{
	#[inline]
	pub fn host(&self) -> Host {
		self.host
	}

	#[inline]
	pub fn backend(&self) -> Backend {
		self.backend
	}

	#[inline]
	pub fn render_kind(&self) -> RenderKind {
		self.render_kind
	}

	#[inline]
	pub fn capabilities(&self) -> HostCapabilities {
		HostCapabilities::new(self.host, self.backend)
	}

	pub fn supports(&self, capability: crate::effect::Capability) -> bool {
		self.capabilities().supports(capability)
	}

	#[inline]
	pub fn layer_width(&self) -> u32 {
		self.layer_width
	}

	#[inline]
	pub fn layer_height(&self) -> u32 {
		self.layer_height
	}

	#[inline]
	pub fn output_width(&self) -> u32 {
		self.output_width
	}

	#[inline]
	pub fn output_height(&self) -> u32 {
		self.output_height
	}

	#[inline]
	pub fn frame_index(&self) -> u32 {
		self.frame_index
	}

	#[inline]
	pub fn time_seconds(&self) -> f32 {
		self.time_seconds
	}

	#[inline]
	pub fn progress(&self) -> f32 {
		self.progress
	}

	pub fn float(&self, param: P) -> Result<f32, after_effects::Error> {
		match &self.inner {
			HostBackend::Cpu { params, .. } => params.float(param),
			HostBackend::Gpu { filter, render_params } => Ok(get_param::<f32, _>(filter, param, render_params)),
		}
	}

	pub fn angle(&self, param: P) -> Result<f32, after_effects::Error> {
		match &self.inner {
			HostBackend::Cpu { params, .. } => params.angle(param),
			HostBackend::Gpu { filter, render_params } => Ok(get_param::<f32, _>(filter, param, render_params)),
		}
	}

	pub fn checkbox(&self, param: P) -> Result<bool, after_effects::Error> {
		match &self.inner {
			HostBackend::Cpu { params, .. } => params.checkbox(param),
			HostBackend::Gpu { filter, render_params } => Ok(get_param::<bool, _>(filter, param, render_params)),
		}
	}

	pub fn popup_zero_based(&self, param: P) -> Result<u32, after_effects::Error> {
		match &self.inner {
			// AE PF popups are 1-based; Premiere GPU popups are already 0-based.
			HostBackend::Cpu { params, .. } => Ok((params.popup(param)? as u32).saturating_sub(1)),
			HostBackend::Gpu { filter, render_params } => Ok(get_param::<i32, _>(filter, param, render_params).max(0) as u32),
		}
	}

	pub fn color(&self, param: P) -> Result<Pixel, after_effects::Error> {
		match &self.inner {
			HostBackend::Cpu { params, .. } => {
				let pf: PF_Pixel = params.color(param)?;
				Ok(Pixel::from_pf_pixel(pf))
			}
			HostBackend::Gpu { filter, render_params } => Ok(get_param::<Pixel, _>(filter, param, render_params)),
		}
	}

	/// Returns the point parameter normalised to the layer dimensions
	/// (`(x / layer_width, y / layer_height)`). Premiere GPU values are
	/// already pre-normalised; AE values come back in pixel space and are
	/// divided here.
	pub fn point_pct(&self, param: P) -> Result<(f32, f32), after_effects::Error> {
		match &self.inner {
			HostBackend::Cpu { params, .. } => {
				let (x, y) = params.point(param)?;
				let w = self.layer_width.max(1) as f32;
				let h = self.layer_height.max(1) as f32;
				Ok((x / w, y / h))
			}
			HostBackend::Gpu { filter, render_params } => Ok(get_param::<(f32, f32), _>(filter, param, render_params)),
		}
	}
}

impl<'a, P> FrameDataContext<'a, P>
where
	P: SetupParams + Eq + Hash + Copy + Debug + 'static,
{
	/// Bridge that lets `kernel_params!` generate a `from_context` method
	/// without leaking [`HostBackend`] across crate boundaries. The two
	/// closures correspond to the macro-generated `from_cpu` / `from_gpu`
	/// constructors; the right one runs based on the active host.
	pub fn extract_kernel_params<K, FCpu, FGpu>(&self, from_cpu: FCpu, from_gpu: FGpu) -> Result<K, after_effects::Error>
	where
		FCpu: FnOnce(&Parameters<P>, f32, f32, bool) -> Result<K, after_effects::Error>,
		FGpu: FnOnce(&pr::GpuFilterData, &pr::RenderParams, f32, f32) -> K,
	{
		match &self.inner {
			HostBackend::Cpu { params, is_premiere } => from_cpu(params, self.layer_width as f32, self.layer_height as f32, *is_premiere),
			HostBackend::Gpu { filter, render_params } => Ok(from_gpu(filter, render_params, self.layer_width as f32, self.layer_height as f32)),
		}
	}
}

/// `Effect::expansion` only sees layer dimensions + parameter values; it
/// runs at AE `SmartPreRender` / `FrameSetup` before any output buffer
/// exists. [`ExpansionContext`] is a thin wrapper around the same
/// extractors so author code can stay close to `frame_data`.
pub type ExpansionContext<'a, P> = FrameDataContext<'a, P>;
