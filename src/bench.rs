//! Criterion harness for prgpu CPU kernels.
//!
//! Wires a `.slang` CPU dispatch into a criterion sweep over
//! `(resolutions × pixel formats)` without touching AE/Premiere plumbing.
//! Throughput is reported as `Throughput::Elements(width * height)` so
//! results print in `Mpx/s`. See `vignette/benches/` for a multi-pass example.

#![cfg(feature = "bench")]

use std::ffi::c_void;
use std::time::Duration;

pub use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use crate::cpu::render::{render_cpu_direct, CpuDispatchTileFn};
use crate::types::Configuration;


/// Pixel format of the synthetic bench buffers.
///
/// Layout id is fixed to BGRA (1) because that is what AE and Premiere
/// hand to CPU kernels in practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
	Bgra8,
	/// 16-bit BGRA. Matches AE's `U15` channel range.
	Bgra16,
	/// 32-bit float BGRA. Matches AE's `F32` world type.
	Bgra32f,
}

impl PixelFormat {
	pub const fn bpp(self) -> u32 {
		match self {
			Self::Bgra8 => 4,
			Self::Bgra16 => 8,
			Self::Bgra32f => 16,
		}
	}

	/// Layout tag for `FrameParams::pixel_layout`. Always 1 (BGRA).
	pub const fn layout_id(self) -> u32 {
		1
	}

	/// Half-precision format flag (AE's `U15`, mapped to `is16f` in `Configuration::cpu`).
	pub const fn is16f(self) -> bool {
		matches!(self, Self::Bgra16)
	}

	pub const fn label(self) -> &'static str {
		match self {
			Self::Bgra8 => "bgra8",
			Self::Bgra16 => "bgra16",
			Self::Bgra32f => "bgra32f",
		}
	}

	pub const ALL: &'static [PixelFormat] = &[Self::Bgra8, Self::Bgra16, Self::Bgra32f];
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
	HD720,
	HD1080,
	UHD4K,
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

	pub fn label(self) -> String {
		let (w, h) = self.dims();
		format!("{w}x{h}")
	}

	pub const COMMON: &'static [Resolution] = &[Self::HD720, Self::HD1080, Self::UHD4K];
}


/// Trailing guard bytes appended past every bench buffer; mirrors `crate::cpu::buffer`'s `ALLOC_GUARD_BYTES` so a kernel off-by-one cannot smash the next allocation.
const ALLOC_GUARD_BYTES: usize = 64;

/// Synthetic frame scene with input/output and optional aux buffers, all
/// tightly packed (`pitch_px == width`). Hands out raw pointers and builds
/// `Configuration`s for the common single-pass case.
pub struct Scene {
	pub width: u32,
	pub height: u32,
	pub format: PixelFormat,
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
	/// Allocate a scene at the given resolution and format.
	///
	/// The input is filled with a deterministic non-trivial pattern so denormal
	/// floats and all-zero branches don't skew the measurement.
	pub fn new(resolution: Resolution, format: PixelFormat) -> Self {
		let (width, height) = resolution.dims();
		let bpp = format.bpp() as usize;
		let byte_len = (width as usize) * (height as usize) * bpp;

		let mut input = vec![0u8; byte_len + ALLOC_GUARD_BYTES];
		let mut output = vec![0u8; byte_len + ALLOC_GUARD_BYTES];

		fill_pattern(&mut input[..byte_len], format, width);
		// Pre-touch output pages so the first iteration isn't skewed by page-fault costs.
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

	/// Allocate an aux buffer (tight pitch); returns the index for `Scene::aux_ptr` / `Scene::aux_dims`.
	pub fn alloc_aux(&mut self, width: u32, height: u32) -> usize {
		let bpp = self.format.bpp() as usize;
		let byte_len = (width as usize) * (height as usize) * bpp;
		let mut data = vec![0u8; byte_len + ALLOC_GUARD_BYTES];
		fill_pattern(&mut data[..byte_len], self.format, width);
		let idx = self.aux.len();
		self.aux.push(AuxBuffer { width, height, data });
		idx
	}

	pub fn input_ptr(&self) -> *mut c_void {
		self.input.as_ptr() as *mut c_void
	}

	pub fn output_ptr(&mut self) -> *mut c_void {
		self.output.as_mut_ptr() as *mut c_void
	}

	pub fn aux_ptr(&mut self, idx: usize) -> *mut c_void {
		self.aux[idx].data.as_mut_ptr() as *mut c_void
	}

	pub fn aux_dims(&self, idx: usize) -> (u32, u32) {
		let a = &self.aux[idx];
		(a.width, a.height)
	}

	/// Build a single-pass `Configuration`: `input → output`, matching dims, tight pitch, `time = 0`.
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
			self.format.bpp(),
			self.format.layout_id(),
		)
	}

	/// Run a single-pass dispatch via the tile dispatcher (one FFI call per rayon chunk).
	pub fn dispatch_simple<P: Copy + Sync>(&mut self, dispatch_tile_fn: CpuDispatchTileFn, params: &P) {
		let cfg = self.simple_config();
		// SAFETY: pointers in `cfg` come from `self` and live for the call; the dispatch only reads input and writes output.
		unsafe { render_cpu_direct("bench", &cfg, dispatch_tile_fn, params) };
	}

	/// Run a kernel with a caller-supplied `Configuration` (multi-pass).
	///
	/// # Safety
	/// `cfg`'s buffer pointers must remain valid for the call; obtain them via
	/// `Scene::input_ptr` / `Scene::output_ptr` / `Scene::aux_ptr` on this scene.
	pub fn dispatch_with<P: Copy + Sync>(&self, cfg: &Configuration, dispatch_tile_fn: CpuDispatchTileFn, params: &P) {
		unsafe { render_cpu_direct("bench", cfg, dispatch_tile_fn, params) };
	}
}

/// Deterministic per-format fill that avoids all-zero / denormal regions. No `rand` dep — fully reproducible.
fn fill_pattern(buf: &mut [u8], format: PixelFormat, _width: u32) {
	match format {
		PixelFormat::Bgra8 => {
			for (i, b) in buf.iter_mut().enumerate() {
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
				// Stay in [0.05, 0.95] to avoid denormals and saturation.
				let x = ((i.wrapping_mul(2654435761) >> 8) & 0xFFFF) as f32 / 65535.0;
				*f = 0.05 + 0.90 * x;
			}
		}
	}
}


/// Fluent builder that wires a CPU kernel into a criterion sweep.
///
/// Defaults: `[HD1080] × [Bgra8]`, `sample_size = 30`, `measurement_time = 5 s`,
/// `warm_up_time = 1 s`. Throughput is reported as `Mpx/s`.
/// Per-combo customization uses a concrete `fn` pointer (see `CustomizeFn`) so
/// closures without captures infer their argument types automatically.
pub type CustomizeFn<P> = fn(&mut P, Resolution, PixelFormat);

pub struct KernelBenchmark<P: Copy + Sync + 'static> {
	name: String,
	dispatch_fn: CpuDispatchTileFn,
	user_params: P,
	resolutions: Vec<Resolution>,
	formats: Vec<PixelFormat>,
	customize: Option<CustomizeFn<P>>,
	sample_size: usize,
	measurement_time: Duration,
	warm_up_time: Duration,
}

impl<P: Copy + Sync + 'static> KernelBenchmark<P> {
	/// Create a benchmark for `dispatch_fn`.
	///
	/// `name` becomes the criterion group; individual benches use the `(format, resolution)` tuple as id.
	pub fn new(name: impl Into<String>, dispatch_fn: CpuDispatchTileFn, user_params: P) -> Self {
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

	pub fn resolutions(mut self, v: &[Resolution]) -> Self {
		self.resolutions = v.to_vec();
		self
	}

	pub fn formats(mut self, v: &[PixelFormat]) -> Self {
		self.formats = v.to_vec();
		self
	}

	/// Per-combo hook for the user params.
	///
	/// Receives a mutable copy of the base params, so it cannot affect other points.
	/// Use for radius-vs-resolution scaling or per-format clamps.
	pub fn customize(mut self, f: CustomizeFn<P>) -> Self {
		self.customize = Some(f);
		self
	}

	/// Override criterion's sample size (default 30).
	pub fn sample_size(mut self, n: usize) -> Self {
		self.sample_size = n;
		self
	}

	/// Override criterion's measurement time (default 5 s).
	pub fn measurement_time(mut self, d: Duration) -> Self {
		self.measurement_time = d;
		self
	}

	/// Override criterion's warm-up time (default 1 s).
	pub fn warm_up_time(mut self, d: Duration) -> Self {
		self.warm_up_time = d;
		self
	}

	/// Execute the sweep. Call from a `criterion_group!` / `criterion_main!` binary.
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
						// Stop the optimizer from elimitating the write.
						std::hint::black_box(scene.output_ptr());
					});
				});
			}
		}

		group.finish();
	}
}
