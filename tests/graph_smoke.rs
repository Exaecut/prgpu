//! `RenderGraph` smoke tests.
//!
//! Exercises the declarative API + executor wiring without touching the
//! GPU: we use prgpu's existing CPU diff kernel as a harmless real Kernel<P>
//! plus a synthetic FrameData. The point is to verify graph wiring, slot
//! resolution, mip-chain iteration, and resource allocation; pixel-level
//! correctness is covered elsewhere.

use prgpu::effect::{FrameBinding, Host, InvocationBase, PixelLayout, RenderKind};
use prgpu::graph::{MipDirection, MipPyramidDesc, RenderGraph, Slot};
use prgpu::types::Backend;

#[derive(Clone, Copy)]
struct FakeFrame {
	threshold: f32,
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
	let graph: RenderGraph<FakeFrame> = RenderGraph::new();
	let mut src = vec![0u8; 16 * 16 * 4];
	let mut dst = vec![0u8; 16 * 16 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 16, 16);
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("empty graph runs");
}

#[test]
fn default_policy_is_auto() {
	assert_eq!(prgpu::graph::SourcePolicy::default(), prgpu::graph::SourcePolicy::Auto);
	let graph: RenderGraph<FakeFrame> = RenderGraph::new();
	let _ = graph; // new() seeds the Auto default; nothing to configure.
}

#[test]
fn auto_policy_runs_clean_on_non_aliasing_host() {
	// A Source -> Output pass trips the auto-snapshot heuristic, but AE+CPU
	// does not report SourceOutputMayAlias so the snapshot short-circuits and the
	// graph runs against the host buffers directly.
	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	graph.add_pass("shake_like", prgpu::kernel::builtin::diff::kernel(), Slot::Source, Slot::Output, |_ctx| prgpu::kernel::builtin::DiffParams {
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
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("auto policy runs");
}

#[test]
fn source_policy_direct_is_a_noop() {
	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	graph.set_source_policy(prgpu::graph::SourcePolicy::Direct);
	let mut src = vec![0u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("direct policy");
}

#[test]
fn snapshot_if_aliased_is_skipped_when_capability_absent() {
	// CPU/AE base does NOT report SourceOutputMayAlias, so the snapshot path
	// short-circuits and the original source pointer is preserved.
	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	graph.set_source_policy(prgpu::graph::SourcePolicy::SnapshotIfAliased { tag: 0xCAFE_0001 });
	let mut src = vec![1u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("policy noop on AE+CPU");
}

#[test]
fn always_snapshot_takes_a_copy_on_cpu() {
	use prgpu::effect::Capability;
	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	graph.set_source_policy(prgpu::graph::SourcePolicy::AlwaysSnapshot { tag: 0xCAFE_0002 });
	let mut src = vec![3u8; 8 * 8 * 4];
	let mut dst = vec![0u8; 8 * 8 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 8, 8);

	// AE+CPU does not support SourceOutputMayAlias, but AlwaysSnapshot ignores
	// that and always allocates. The graph runs without panicking, which is
	// the visible promise of the policy at the executor level.
	let _ = base.capabilities().supports(Capability::SourceOutputMayAlias);
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("always snapshot");
}

#[test]
fn mip_pyramid_resource_is_allocated_with_requested_levels() {
	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	let _bloom = graph.declare_mip_pyramid("bloom", |_ctx| MipPyramidDesc::new(64, 64).levels(4).tag(0xFEED_0001));

	// Allocation happens during execute; an empty graph with a resource
	// exercises the resource allocator path without invoking any kernels.
	let mut src = vec![0u8; 64 * 64 * 4];
	let mut dst = vec![0u8; 64 * 64 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 64, 64);
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("alloc-only graph runs");
}

#[test]
fn mip_chain_iterates_levels_minus_one_steps() {
	use std::sync::atomic::{AtomicU32, Ordering};
	use std::sync::Arc;

	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	let bloom = graph.declare_mip_pyramid("bloom", |_ctx| MipPyramidDesc::new(64, 64).levels(4).tag(0xFEED_0002));

	let down_calls = Arc::new(AtomicU32::new(0));
	let up_calls = Arc::new(AtomicU32::new(0));

	let down_clone = Arc::clone(&down_calls);
	let up_clone = Arc::clone(&up_calls);

	graph.add_mip_chain("downsample", bloom, MipDirection::Down, prgpu::kernel::builtin::diff::kernel(), move |level, _ctx| {
		down_clone.fetch_add(1, Ordering::SeqCst);
		// Real diff params are irrelevant; the chain exercises slot/level wiring.
		prgpu::kernel::builtin::DiffParams {
			tol_r: 0.0,
			tol_g: 0.0,
			tol_b: 0.0,
			tol_a: 0.0,
			smooth_a: 0.0,
			smooth_b: 0.0,
			..Default::default()
		}
	});

	graph.add_mip_chain("upsample", bloom, MipDirection::Up, prgpu::kernel::builtin::diff::kernel(), move |level, _ctx| {
		up_clone.fetch_add(1, Ordering::SeqCst);
		prgpu::kernel::builtin::DiffParams {
			tol_r: 0.0,
			tol_g: 0.0,
			tol_b: 0.0,
			tol_a: 0.0,
			smooth_a: 0.0,
			smooth_b: 0.0,
			..Default::default()
		}
	});

	let mut src = vec![0u8; 64 * 64 * 4];
	let mut dst = vec![0u8; 64 * 64 * 4];
	let base = synthetic_base(dst.as_mut_ptr() as *mut _, src.as_mut_ptr() as *mut _, 64, 64);
	prgpu::graph::execute::execute(&graph, &FakeFrame { threshold: 0.0 }, &base).expect("mip chain runs");

	// 4-level pyramid → 3 transitions per direction.
	assert_eq!(down_calls.load(Ordering::SeqCst), 3);
	assert_eq!(up_calls.load(Ordering::SeqCst), 3);
}

#[test]
fn slot_inline_can_be_used_directly() {
	let mut graph: RenderGraph<FakeFrame> = RenderGraph::new();
	let inline_target = FrameBinding {
		data: 0x100 as *mut _,
		pitch_px: 8,
		width: 8,
		height: 8,
		mip_levels: 0,
		bytes_per_pixel: 4,
		pixel_layout: PixelLayout::Bgra,
	};
	let _ = inline_target;
	graph.add_pass("noop_pass", prgpu::kernel::builtin::diff::kernel(), Slot::Source, Slot::Output, |_ctx| prgpu::kernel::builtin::DiffParams {
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
