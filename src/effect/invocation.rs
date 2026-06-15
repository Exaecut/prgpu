//! Normalised per-render invocation context.
//!
//! Adobe hands every render path a different bag of host objects (AE PF
//! `InData` + layers, Premiere `GpuFilterData` + `RenderParams` + PPix
//! handles, the prgpu test harness, ...). The adapters extract those into a
//! single [`InvocationBase`] before the graph executor or `ConfigBuilder`
//! sees anything. Effect code never touches the raw host objects unless it
//! opts into a hook.
//!
//! `InvocationBase` is host-side metadata; it does not own pixel buffers.
//! Pointers it carries (`device_handle`, frame data) follow the same ABI
//! contract as [`crate::types::Configuration`].

use std::ffi::c_void;

use crate::effect::host::{Host, RenderKind};
use crate::types::Backend;

/// Pixel layout id matching the `vekl::Layout` slang enum and the integer
/// codes the kernels consume via `FrameParams.{out,in,dst}_desc.layout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PixelLayout {
	Rgba = 0,
	Bgra = 1,
	Vuya601 = 2,
	Vuya709 = 3,
}

impl PixelLayout {
	pub const fn as_u32(self) -> u32 {
		self as u32
	}

	pub const fn from_u32(v: u32) -> Self {
		match v {
			0 => PixelLayout::Rgba,
			2 => PixelLayout::Vuya601,
			3 => PixelLayout::Vuya709,
			_ => PixelLayout::Bgra,
		}
	}
}

/// One source-of-truth view over an Adobe / test pixel buffer.
///
/// Width / height describe the buffer extent (not necessarily the dispatch
/// extent — that's controlled by the pass via `ConfigBuilder::dispatch_size`).
/// Pitch is in pixels for parity with `Configuration::*_pitch_px`.
#[derive(Debug, Clone, Copy)]
pub struct FrameBinding {
	pub data: *mut c_void,
	pub pitch_px: i32,
	pub width: u32,
	pub height: u32,
	pub mip_levels: u32,
	pub bytes_per_pixel: u32,
	pub pixel_layout: PixelLayout,
}

// `data` is a host-owned pixel pointer that lives across the dispatch and is
// only ever read/written via the host's GPU context (or the rayon CPU pool's
// scoped pointer copy). Same contract as `crate::types::Configuration`.
unsafe impl Send for FrameBinding {}
unsafe impl Sync for FrameBinding {}

/// Maximum number of secondary image inputs (AE layer params / Premiere
/// track-matte frames) an effect may bind through [`crate::graph::pass::Slot::Layer`].
/// Kept small and fixed so [`InvocationBase`] stays `Copy`-cloneable without a
/// heap allocation; bump alongside any effect that needs more aux layers.
pub const MAX_AUX_LAYERS: usize = 4;

impl FrameBinding {
	pub const fn null(bytes_per_pixel: u32, pixel_layout: PixelLayout) -> Self {
		Self {
			data: std::ptr::null_mut(),
			pitch_px: 0,
			width: 0,
			height: 0,
			mip_levels: 0,
			bytes_per_pixel,
			pixel_layout,
		}
	}

	pub fn is_null(&self) -> bool {
		self.data.is_null()
	}
}

/// Per-render normalised state. Built once by the adapter, reused for every
/// pass via [`crate::types::ConfigBuilder`]. Not `Send + Sync` because of
/// the raw pointers — same contract as [`crate::types::Configuration`].
pub struct InvocationBase {
	pub host: Host,
	pub backend: Backend,
	pub render_kind: RenderKind,

	pub device_handle: *mut c_void,
	pub context_handle: Option<*mut c_void>,
	pub command_queue_handle: *mut c_void,

	pub bytes_per_pixel: u32,
	pub pixel_layout: PixelLayout,
	/// Vekl `PixelStorage` tag (0=Unorm8x4, 1=Unorm16x4, 2=Float32x4, 3=Float16x4).
	/// Set by the adapter from the host pixel format; carried into every pass's
	/// `Configuration` so half-float GPU buffers decode correctly.
	pub storage: u32,
	/// 0 = top-down; 1 = bottom-up host buffer (Premiere CPU). Applied uniformly to
	/// every buffer access so kernel UV is top-left on all backends.
	pub flip_y: u32,
	pub time: f32,
	pub progress: f32,
	pub render_generation: u64,

	/// Source top-left offset inside the output canvas. (0,0) when output == source.
	pub ext_x: i32,
	pub ext_y: i32,

	pub source: FrameBinding,
	/// Secondary image inputs resolved by the adapter (AE layer params via
	/// `checkout_layer_pixels`; Premiere leaves these `None` — layer params are
	/// inert there). Indexed by the per-effect layer-param order, which the
	/// `params!` macro exposes as `<Marker>::LAYER_INDEX` and the pipeline binds
	/// via [`crate::graph::pass::Slot::Layer`]. A `None` slot means "not
	/// assigned / checkout failed"; pipelines fall back to `source`.
	pub layers: [Option<FrameBinding>; MAX_AUX_LAYERS],
	pub output: FrameBinding,
}

impl InvocationBase {
	pub fn capabilities(&self) -> crate::effect::host::HostCapabilities {
		crate::effect::host::HostCapabilities::new(self.host, self.backend)
	}

	/// Per-slot presence flags derived from the actual checkout result. This is
	/// the source of truth a pipeline's `Ctx::layer_present` reflects, so a
	/// layer the host failed to deliver is treated as absent (fallback to
	/// `source`) rather than read as garbage.
	pub fn layer_presence(&self) -> [bool; MAX_AUX_LAYERS] {
		let mut out = [false; MAX_AUX_LAYERS];
		for (i, slot) in self.layers.iter().enumerate() {
			out[i] = slot.map(|b| !b.is_null()).unwrap_or(false);
		}
		out
	}
}
