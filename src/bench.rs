//! DX-friendly criterion harness for benchmarking CPU kernels.
//!
//! # Why this exists
//!
//! `prgpu::bench` gives kernel developers a **worry-less** way to wire a
//! `.vekl`-backed CPU kernel into a `criterion` benchmark without touching any
//! After Effects / Premiere plumbing.
//!
//! It provides:
//! - [`PixelFormat`] / [`Resolution`] enums with sensible defaults
//! - [`Scene`] — owns synthetic input / output / auxiliary buffers and knows how
//!   to build a [`Configuration`] for a single-pass dispatch
//! - [`KernelBenchmark`] — fluent builder that runs a full
//!   `(resolutions × formats)` sweep against any [`CpuDispatchFn`]
//! - Re-exports of the `criterion` entry points so a bench file only needs to
//!   import `prgpu::bench::*`
//!
//! # Minimal example
//!
//! ```ignore
//! use prgpu::bench::*;
//!
//! fn bench_my_kernel(c: &mut Criterion) {
//!     KernelBenchmark::new(
//!         "my_kernel",
//!         my_kernel::MY_KERNEL_CPU_DISPATCH,
//!         MyParams::default(),
//!     )
//!     .resolutions(&[Resolution::HD1080, Resolution::UHD4K])
//!     .formats(&[PixelFormat::Bgra8, PixelFormat::Bgra32f])
//!     .run(c);
//! }
//!
//! criterion_group!(benches, bench_my_kernel);
//! criterion_main!(benches);
//! ```
//!
//! # Multi-pass kernels
//!
//! For kernels that need an intermediate buffer (e.g. separable blur), build
//! the [`Configuration`]s yourself with [`Scene::input_ptr`], [`Scene::aux_ptr`],
//! [`Scene::output_ptr`], and drive them via [`Scene::dispatch_with`] inside a
//! criterion `b.iter(...)` closure. See the `vignette/benches/` example.
//!
//! # Statistical rules of thumb (defaults)
//!
//! - `sample_size = 30`  (enough for CV < 3 % on stable workloads)
//! - `measurement_time = 5 s`
//! - `warm_up_time = 1 s`
//! - Throughput is reported as `Throughput::Elements(width * height)` so the
//!   "Mpx/s" figure is directly comparable across resolutions.

#![cfg(feature = "bench")]

use std::ffi::c_void;
use std::time::Duration;

pub use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use crate::cpu::render::{render_cpu_direct, CpuDispatchFn};
use crate::types::Configuration;

// ---------------------------------------------------------------------------
// PixelFormat
// ---------------------------------------------------------------------------

/// Pixel format of the synthetic buffers used by the benchmark.
///
/// The `layout_id` is fixed to BGRA (`1`) because that is what both AE and
/// Premiere hand to CPU kernels in practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
	/// 8-bit BGRA, `bpp = 4`.
	Bgra8,
	/// 16-bit BGRA, `bpp = 8`. Semantically matches AE's U15 channel range.
	Bgra16,
	/// 32-bit float BGRA, `bpp = 16`. Matches AE's `F32` world type.
	Bgra32f,
}

impl PixelFormat {
	/// Bytes per pixel as seen by the kernel.
	pub const fn bpp(self) -> u32 {
		match self {
			Self::Bgra8 => 4,
			Self::Bgra16 => 8,
			Self::Bgra32f => 16,
		}
	}

	/// Layout tag passed in `FrameParams::pixel_layout`. BGRA (`1`) in all
	/// current cases.
	pub const fn layout_id(self) -> u32 {
		1
	}

	/// Whether the backing format is half-precision (AE's `U15`, mapped to
	/// `is16f` in [`Configuration::cpu`]).
	pub const fn is16f(self) -> bool {
		matches!(self, Self::Bgra16)
	}

	/// Short human-readable tag (used in bench IDs).
	pub const fn label(self) -> &'static str {
		match self {
			Self::Bgra8 => "bgra8",
			Self::Bgra16 => "bgra16",
			Self::Bgra32f => "bgra32f",
		}
	}

	/// All three formats, in the order `[u8, u16, f32]`.
	pub const ALL: &'static [PixelFormat] = &[Self::Bgra8, Self::Bgra16, Self::Bgra32f];
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Output resolution of the synthetic scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
	/// 1280×720
	HD720,
	/// 1920×1080
	HD1080,
	/// 3840×2160
	UHD4K,
	/// Arbitrary (`width`, `height`).
	Custom(u32, u32),
}

impl Resolution {
	pub const fn dims(self) -> (u32, u32) {
		match self {
			Self::HD720 => (1280, 720),
			Self::HD1080 => (1920, 1080),
			Self::UHD4K => (3840, 2160),
			Self::Custom(w, h) => (w, h),
		}
	}

	/// Short human-readable tag (used in bench IDs).
	pub fn label(self) -> String {
		let (w, h) = self.dims();
		format!("{w}x{h}")
	}

	/// Convenience set for quick sweeps: `[HD720, HD1080, UHD4K]`.
	pub const COMMON: &'static [Resolution] = &[Self::HD720, Self::HD1080, Self::UHD4K];
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// Extra bytes appended past the nominal end of every bench buffer, matching
/// the `ALLOC_GUARD_BYTES` semantics in [`crate::cpu::buffer`]. Prevents heap
/// corruption if a kernel off-by-ones into the next allocation.
const ALLOC_GUARD_BYTES: usize = 64;

/// Synthetic frame scene. Owns:
///
/// - an **input** buffer (`width × height × bpp` bytes), deterministically
///   pre-filled
/// - an **output** buffer (same geometry)
/// - zero or more **auxiliary** buffers for multi-pass kernels (e.g. the
///   intermediate of a separable blur)
///
/// All buffers are tightly packed — `pitch_px == width`.
///
/// `Scene` is the canonical "workspace" passed to a benchmarked kernel. It
/// knows how to hand out raw pointers and build [`Configuration`]s for the
/// common single-pass case.
pub struct Scene {
	pub width: u32,
	pub height: u32,
	pub format: PixelFormat,
	/// Pitch in pixels. Tightly packed, equal to `width`.
	pub pitch_px: i32,
	input: Vec<u8>,
	output: Vec<u8>,
	aux: Vec<AuxBuffer>,
}

struct AuxBuffer {
	width: u32,
	height: u32,
	data: Vec<u8>,
}

impl Scene {
	/// Allocate a new scene with the given resolution and pixel format.
	///
	/// The input buffer is pre-filled with a deterministic, non-trivial
	/// pattern to avoid denormal floats and all-zero branches skewing the
	/// measurement.
	pub fn new(resolution: Resolution, format: PixelFormat) -> Self {
		let (width, height) = resolution.dims();
		let bpp = format.bpp() as usize;
		let byte_len = (width as usize) * (height as usize) * bpp;

		let mut input = vec![0u8; byte_len + ALLOC_GUARD_BYTES];
		let mut output = vec![0u8; byte_len + ALLOC_GUARD_BYTES];

		fill_pattern(&mut input[..byte_len], format, width);
		// Pre-touch output pages so the first iteration isn't skewed by
		// page-fault / first-touch costs.
		fill_pattern(&mut output[..byte_len], format, width);

		Self {
			width,
			height,
			format,
			pitch_px: width as i32,
			input,
			output,
			aux: Vec::new(),
		}
	}

	/// Allocate an auxiliary buffer with the given dimensions (tight pitch).
	/// Returns the aux index usable with [`Scene::aux_ptr`] / [`Scene::aux_dims`].
	pub fn alloc_aux(&mut self, width: u32, height: u32) -> usize {
		let bpp = self.format.bpp() as usize;
		let byte_len = (width as usize) * (height as usize) * bpp;
		let mut data = vec![0u8; byte_len + ALLOC_GUARD_BYTES];
		fill_pattern(&mut data[..byte_len], self.format, width);
		let idx = self.aux.len();
		self.aux.push(AuxBuffer { width, height, data });
		idx
	}

	/// Raw pointer to the input buffer (bench-scope lifetime).
	pub fn input_ptr(&self) -> *mut c_void {
		self.input.as_ptr() as *mut c_void
	}

	/// Raw pointer to the output buffer (bench-scope lifetime).
	pub fn output_ptr(&mut self) -> *mut c_void {
		self.output.as_mut_ptr() as *mut c_void
	}

	/// Raw pointer to an auxiliary buffer.
	pub fn aux_ptr(&mut self, idx: usize) -> *mut c_void {
		self.aux[idx].data.as_mut_ptr() as *mut c_void
	}

	/// `(width, height)` of an auxiliary buffer.
	pub fn aux_dims(&self, idx: usize) -> (u32, u32) {
		let a = &self.aux[idx];
		(a.width, a.height)
	}

	/// Build a single-pass `Configuration`: `input → output`, matching sizes,
	/// tight pitch.
	pub fn simple_config(&mut self) -> Configuration {
		let ptr_in = self.input.as_ptr() as *mut c_void;
		let ptr_out = self.output.as_mut_ptr() as *mut c_void;
		Configuration::cpu(
			ptr_in,
			ptr_out,
			self.pitch_px,
			self.pitch_px,
			self.width,
			self.height,
			self.format.is16f(),
			self.format.bpp(),
			self.format.layout_id(),
		)
	}

	/// Run a single-pass dispatch (`input → output`) on this scene.
	pub fn dispatch_simple<P: Copy + Sync>(&mut self, dispatch_fn: CpuDispatchFn, params: &P) {
		let cfg = self.simple_config();
		// SAFETY: pointers in `cfg` come from `self` and live for the call;
		// the dispatch function only reads `input` and writes `output`.
		unsafe { render_cpu_direct("bench", &cfg, dispatch_fn, params) };
	}

	/// Run a kernel with a user-supplied [`Configuration`] (multi-pass scenario).
	///
	/// # Safety
	/// `cfg`'s buffer pointers must remain valid for the duration of the call;
	/// typically obtained from [`Scene::input_ptr`] / [`Scene::output_ptr`] /
	/// [`Scene::aux_ptr`] on this same [`Scene`].
	pub fn dispatch_with<P: Copy + Sync>(&self, cfg: &Configuration, dispatch_fn: CpuDispatchFn, params: &P) {
		unsafe { render_cpu_direct("bench", cfg, dispatch_fn, params) };
	}
}

/// Deterministic fill: varies by x/y and format, avoids all-zero / denormal
/// regions. Does not use `rand` on purpose (no extra dep, fully reproducible).
fn fill_pattern(buf: &mut [u8], format: PixelFormat, _width: u32) {
	match format {
		PixelFormat::Bgra8 => {
			for (i, b) in buf.iter_mut().enumerate() {
				// Simple LCG-ish pattern, range [1, 255].
				let v = (i.wrapping_mul(2654435761) >> 24) as u8;
				*b = v.max(1);
			}
		}
		PixelFormat::Bgra16 => {
			let words = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u16, buf.len() / 2) };
			for (i, w) in words.iter_mut().enumerate() {
				let v = (i.wrapping_mul(2654435761) >> 16) as u16;
				*w = v.max(1);
			}
		}
		PixelFormat::Bgra32f => {
			let floats = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, buf.len() / 4) };
			for (i, f) in floats.iter_mut().enumerate() {
				// Produce values in [0.05, 0.95] to avoid denormals and saturation.
				let x = ((i.wrapping_mul(2654435761) >> 8) & 0xFFFF) as f32 / 65535.0;
				*f = 0.05 + 0.90 * x;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// KernelBenchmark builder
// ---------------------------------------------------------------------------

/// Fluent builder that wires a CPU kernel into a `criterion` sweep.
///
/// Designed so a kernel developer can get a statistically-meaningful benchmark
/// with ~5 lines. See the module docs for the canonical example.
///
/// Default configuration:
/// - sweep over `[Resolution::HD1080]` × `[PixelFormat::Bgra8]`
/// - `sample_size = 30`, `measurement_time = 5 s`, `warm_up_time = 1 s`
///
/// The throughput is automatically reported as `Throughput::Elements(w*h)`
/// so criterion prints `Mpx/s`.
/// Signature for the per-combo customization hook passed to
/// [`KernelBenchmark::customize`]. A plain `fn` pointer is used (rather than
/// a generic `Fn` trait bound) so that closure argument types are inferred
/// directly from this concrete signature — users can write
/// `|p, res, fmt| { ... }` without annotating the parameters.
pub type CustomizeFn<P> = fn(&mut P, Resolution, PixelFormat);

pub struct KernelBenchmark<P: Copy + Sync + 'static> {
	name: String,
	dispatch_fn: CpuDispatchFn,
	user_params: P,
	resolutions: Vec<Resolution>,
	formats: Vec<PixelFormat>,
	customize: Option<CustomizeFn<P>>,
	sample_size: usize,
	measurement_time: Duration,
	warm_up_time: Duration,
}

impl<P: Copy + Sync + 'static> KernelBenchmark<P> {
	/// Create a new benchmark for `dispatch_fn`. `name` becomes the criterion
	/// **group** name; individual benches within the group are identified by
	/// the `(format, resolution)` tuple.
	pub fn new(name: impl Into<String>, dispatch_fn: CpuDispatchFn, user_params: P) -> Self {
		Self {
			name: name.into(),
			dispatch_fn,
			user_params,
			resolutions: vec![Resolution::HD1080],
			formats: vec![PixelFormat::Bgra8],
			customize: None,
			sample_size: 30,
			measurement_time: Duration::from_secs(5),
			warm_up_time: Duration::from_secs(1),
		}
	}

	/// Override the resolution sweep.
	pub fn resolutions(mut self, v: &[Resolution]) -> Self {
		self.resolutions = v.to_vec();
		self
	}

	/// Override the pixel-format sweep.
	pub fn formats(mut self, v: &[PixelFormat]) -> Self {
		self.formats = v.to_vec();
		self
	}

	/// Per-combo customization of the user params.
	///
	/// Useful when some parameter depends on the scene — e.g. a blur radius
	/// that must scale with resolution, or a per-format clamp. The hook
	/// receives a mutable reference to a **copy** of the base params, so it
	/// cannot affect other bench points.
	///
	/// Uses a concrete `fn` pointer (see [`CustomizeFn`]) so closures without
	/// captures infer their argument types automatically — you can write
	/// `|p, res, fmt| { ... }` with no type annotations.
	pub fn customize(mut self, f: CustomizeFn<P>) -> Self {
		self.customize = Some(f);
		self
	}

	/// Override criterion's sample size (default: 30).
	pub fn sample_size(mut self, n: usize) -> Self {
		self.sample_size = n;
		self
	}

	/// Override criterion's measurement time (default: 5 s).
	pub fn measurement_time(mut self, d: Duration) -> Self {
		self.measurement_time = d;
		self
	}

	/// Override criterion's warm-up time (default: 1 s).
	pub fn warm_up_time(mut self, d: Duration) -> Self {
		self.warm_up_time = d;
		self
	}

	/// Execute the full sweep. Must be called from a `#[bench]`-free binary
	/// using the `criterion_group!` / `criterion_main!` pair.
	pub fn run(self, c: &mut Criterion) {
		let dispatch_fn = self.dispatch_fn;
		let base_params = self.user_params;
		let customize = self.customize;
		let sample_size = self.sample_size;
		let m_time = self.measurement_time;
		let w_time = self.warm_up_time;

		let mut group = c.benchmark_group(&self.name);
		group.sample_size(sample_size);
		group.measurement_time(m_time);
		group.warm_up_time(w_time);

		for &fmt in &self.formats {
			for &res in &self.resolutions {
				let (w, h) = res.dims();
				let pixels = (w as u64) * (h as u64);
				group.throughput(Throughput::Elements(pixels));

				let mut params = base_params;
				if let Some(f) = customize {
					f(&mut params, res, fmt);
				}

				let id = BenchmarkId::new(fmt.label(), res.label());
				group.bench_function(id, move |b| {
					let mut scene = Scene::new(res, fmt);
					b.iter(|| {
						scene.dispatch_simple(dispatch_fn, &params);
						// Prevent the optimizer from eliminating the write.
						std::hint::black_box(scene.output_ptr());
					});
				});
			}
		}

		group.finish();
	}
}
