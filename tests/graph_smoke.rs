//! `Graph` smoke tests.
//!
//! Exercises the declarative API + executor wiring without touching the
//! GPU: we use prgpu's existing CPU diff kernel as a harmless real Kernel<P>
//! plus a synthetic Ctx. The point is to verify graph wiring, slot
//! resolution, mip-chain iteration, and resource allocation; pixel-level
//! correctness is covered elsewhere.

use std::fmt::Debug;
use std::hash::Hash;

use after_effects::Parameters;
use premiere as pr;
use prgpu::effect::ctx::{Ctx, Geometry, Timing};
use prgpu::effect::host::{Host, HostCapabilities};
use prgpu::effect::{FrameBinding, InvocationBase, PixelLayout, RenderKind};
use prgpu::graph::{Graph, MipDirection, MipPyramidDesc, Slot, SourcePolicy};
use prgpu::params::{Color, FromParamValue, Param, ParamValue, ParamsSpec, Point2, Snapshot, SnapshotGeom};
use prgpu::types::Backend;

/// Minimal synthetic ParamsSpec — no real params, just enough to compile.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(usize)]
enum FakeParams { _None = 1 }

impl From<FakeParams> for usize {
	fn from(p: FakeParams) -> usize { p as usize }
}

#[derive(Clone, Copy)]
struct FakeSnapshot;
impl Default for FakeSnapshot { fn default() -> Self { Self } }
impl Snapshot<FakeParams> for FakeSnapshot {
	fn value(&self, _id: FakeParams) -> ParamValue { ParamValue::None }
	fn set(&mut self, _id: FakeParams, _value: ParamValue) {}
}

impl ParamsSpec for FakeParams {
	const COUNT: usize = 1;
	const DEBUG_PARAM: Option<Self> = None;
	type Snapshot = FakeSnapshot;
	fn register(_params: &mut Parameters<Self>) -> Result<(), after_effects::Error> { Ok(()) }
	fn snapshot_cpu(_params: &Parameters<Self>, _geom: &SnapshotGeom) -> Result<Self::Snapshot, after_effects::Error> { Ok(FakeSnapshot) }
	fn snapshot_gpu(_filter: &pr::GpuFilterData, _rp: &pr::RenderParams, _geom: &SnapshotGeom) -> Self::Snapshot { FakeSnapshot }
	fn buttons() -> &'static [(Self, fn())] { &[] }
}

fn synthetic_ctx() -> Ctx<'static, FakeParams> {
	static SNAPSHOT: std::sync::LazyLock<FakeSnapshot> = std::sync::LazyLock::new(|| FakeSnapshot);
	let geom = Geometry { layer_w: 64, layer_h: 64, output_w: 64, output_h: 64, ext_x: 0, ext_y: 0 };
	let caps = HostCapabilities::new(Host::AfterEffects, Backend::Cpu);
	let time = Timing { frame_index: 0, time_seconds: 0.0, progress: 0.0 };
	// Build and return — Ctx borrows from the static snapshot.
	Ctx::<FakeParams>::new(&*SNAPSHOT, geom, time, caps, false)
}

fn synthetic_base(out_data: *mut std::ffi::c_void, src_data: *mut std::ffi::c_void, w: u32, h: u32) -> InvocationBase {
	let source = FrameBinding {
		data: src_data,
		pitch_px: w as i32,
		width: w,
		height: h,
		mip_levels: 0,
		bytes_per_pixel: 4,
		pixel_layout: PixelLayout::Bgra,
	};
	let output = FrameBinding {
		data: out_data,
		pitch_px: w as i32,
		width: w,
		height: h,
		mip_levels: 0,
		bytes_per_pixel: 4,
		pixel_layout: PixelLayout::Bgra,
	};
	InvocationBase {
		host: Host::AfterEffects,
		backend: Backend::Cpu,
		render_kind: RenderKind::TestCpu,
		device_handle: std::ptr::null_mut(),
		context_handle: None,
		command_queue_handle: std::ptr::null_mut(),
		bytes_per_pixel: 4,
		pixel_layout: PixelLayout::Bgra,
		storage: 0,
		flip_y: 0,
		time: 0.0,
		progress: 0.0,
		render_generation: 0,
		ext_x: 0,
		ext_y: 0,
		source,
		secondary_source: None,
		output,
	}
}

#[test]
fn empty_graph_runs_clean() {
	let graph: Graph<FakeParams> = Graph::new();
	let mut src = vec![0u8; 16 * 16 * 4];
	let mut dst = vec![0u8; 16 * 16 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 16, 16);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("empty graph runs");
}

#[test]
fn default_policy_is_auto() {
	assert_eq!(SourcePolicy::default(), SourcePolicy::Auto);
}

#[test]
fn auto_policy_runs_clean_on_non_aliasing_host() {
	let mut graph: Graph<FakeParams> = Graph::new();
	graph.pass_with(prgpu::kernel::builtin::diff::kernel(), |_ctx| prgpu::kernel::builtin::DiffParams {
		tol_r: 0.0,
		tol_g: 0.0,
		tol_b: 0.0,
		tol_a: 0.0,
		smooth_a: 0.0,
		smooth_b: 0.0,
		..Default::default()
	});
	let mut src = vec![2u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("auto policy runs");
}

#[test]
fn source_policy_direct_is_a_noop() {
	let mut graph: Graph<FakeParams> = Graph::new();
	graph.source_policy(SourcePolicy::Direct);
	let mut src = vec![0u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("direct policy");
}

#[test]
fn snapshot_if_aliased_is_skipped_when_capability_absent() {
	let mut graph: Graph<FakeParams> = Graph::new();
	graph.source_policy(SourcePolicy::SnapshotIfAliased { tag: 0xCAFE_0001 });
	let mut src = vec![1u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("policy noop on AE+CPU");
}

#[test]
fn always_snapshot_takes_a_copy_on_cpu() {
	let mut graph: Graph<FakeParams> = Graph::new();
	graph.source_policy(SourcePolicy::AlwaysSnapshot { tag: 0xCAFE_0002 });
	let mut src = vec![3u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("always snapshot");
}

#[test]
fn mip_pyramid_resource_is_allocated_with_requested_levels() {
	let mut graph: Graph<FakeParams> = Graph::new();
	let _bloom = graph.mip_pyramid("bloom", |_ctx| MipPyramidDesc::new(64, 64).levels(4).tag(0xFEED_0001));
	let mut src = vec![0u8; 64 * 64 * 4];
	let mut dst = vec![0u8; 64 * 64 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 64, 64);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("alloc-only graph runs");
}

#[test]
fn mip_chain_iterates_levels_minus_one_steps() {
	use std::sync::atomic::{AtomicU32, Ordering};
	use std::sync::Arc;

	let mut graph: Graph<FakeParams> = Graph::new();
	let bloom = graph.mip_pyramid("bloom", |_ctx| MipPyramidDesc::new(64, 64).levels(4).tag(0xFEED_0002));

	let down_calls = Arc::new(AtomicU32::new(0));
	let up_calls = Arc::new(AtomicU32::new(0));

	let down_clone = Arc::clone(&down_calls);
	let up_clone = Arc::clone(&up_calls);

	graph.mip_chain(bloom, MipDirection::Down, prgpu::kernel::builtin::diff::kernel())
		.params(move |level, _ctx| {
			down_clone.fetch_add(1, Ordering::SeqCst);
			prgpu::kernel::builtin::DiffParams {
				tol_r: 0.0, tol_g: 0.0, tol_b: 0.0, tol_a: 0.0,
				smooth_a: 0.0, smooth_b: 0.0,
				..Default::default()
			}
		});

	graph.mip_chain(bloom, MipDirection::Up, prgpu::kernel::builtin::diff::kernel())
		.params(move |level, _ctx| {
			up_clone.fetch_add(1, Ordering::SeqCst);
			prgpu::kernel::builtin::DiffParams {
				tol_r: 0.0, tol_g: 0.0, tol_b: 0.0, tol_a: 0.0,
				smooth_a: 0.0, smooth_b: 0.0,
				..Default::default()
			}
		});

	let mut src = vec![0u8; 64 * 64 * 4];
	let mut dst = vec![0u8; 64 * 64 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 64, 64);
	let ctx = synthetic_ctx();
	prgpu::graph::execute::execute(&graph, &ctx, &base).expect("mip chain runs");

	assert_eq!(down_calls.load(Ordering::SeqCst), 3);
	assert_eq!(up_calls.load(Ordering::SeqCst), 3);
}

#[test]
fn slot_inline_can_be_used_directly() {
	let mut graph: Graph<FakeParams> = Graph::new();
	graph.pass_with(prgpu::kernel::builtin::diff::kernel(), |_ctx| prgpu::kernel::builtin::DiffParams {
		tol_r: 0.0,
		tol_g: 0.0,
		tol_b: 0.0,
		tol_a: 0.0,
		smooth_a: 0.0,
		smooth_b: 0.0,
		..Default::default()
	});
	assert_eq!(graph.pass_count(), 1);
}
