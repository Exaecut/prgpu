//! `ConfigBuilder` round-trip tests against a synthetic [`InvocationBase`].
//!
//! Constructs a fake CPU invocation with main-source / output bindings and
//! verifies the produced [`Configuration`] matches what the legacy manual
//! `Configuration::cpu` builder would have emitted (pitches, dimensions,
//! pointers, mip levels, pixel layout).

use prgpu::effect::{FrameBinding, Host, InvocationBase, PixelLayout, RenderKind};
use prgpu::types::{Backend, ConfigBuilder, ConfigBuildError, PassBinding};

fn make_test_base() -> InvocationBase {
	let main = FrameBinding {
		data: 0x1000 as *mut _,
		pitch_px: 1920,
		width: 1920,
		height: 1080,
		mip_levels: 0,
		bytes_per_pixel: 4,
		pixel_layout: PixelLayout::Bgra,
	};
	let output = FrameBinding {
		data: 0x2000 as *mut _,
		pitch_px: 1920,
		width: 1920,
		height: 1080,
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
		time: 0.5,
		progress: 0.25,
		render_generation: 7,
		ext_x: 0,
		ext_y: 0,
		main_source: main,
		incoming_source: None,
		outgoing_source: None,
		output,
	}
}

#[test]
fn source_to_output_pass() {
	let base = make_test_base();
	let cfg = ConfigBuilder::new(&base).source(PassBinding::MainSource).target(PassBinding::Output).build().expect("builds");
	assert_eq!(cfg.dest_data as usize, 0x2000);
	assert_eq!(cfg.outgoing_data.unwrap() as usize, 0x1000);
	assert_eq!(cfg.width, 1920);
	assert_eq!(cfg.height, 1080);
	assert_eq!(cfg.outgoing_pitch_px, 1920);
	assert_eq!(cfg.dest_pitch_px, 1920);
	assert_eq!(cfg.bytes_per_pixel, 4);
	assert_eq!(cfg.pixel_layout, 1);
}

#[test]
fn dispatch_size_overrides_dest_dims() {
	let base = make_test_base();
	let cfg = ConfigBuilder::new(&base).source(PassBinding::MainSource).target(PassBinding::Output).dispatch_size(960, 540).build().expect("builds");
	assert_eq!(cfg.width, 960);
	assert_eq!(cfg.height, 540);
	assert_eq!(cfg.outgoing_width, 1920);
}

#[test]
fn missing_dest_is_rejected() {
	let base = make_test_base();
	let res = ConfigBuilder::new(&base).source(PassBinding::MainSource).build();
	assert_eq!(res.unwrap_err(), ConfigBuildError::MissingDest);
}

#[test]
fn zero_dispatch_size_is_rejected() {
	let base = make_test_base();
	let res = ConfigBuilder::new(&base).target(PassBinding::Output).dispatch_size(0, 540).build();
	assert_eq!(res.unwrap_err(), ConfigBuildError::ZeroDispatchSize);
}

#[test]
fn mip_levels_are_propagated() {
	let mut base = make_test_base();
	let pyramid_buf = FrameBinding {
		data: 0x3000 as *mut _,
		pitch_px: 960,
		width: 960,
		height: 540,
		mip_levels: 5,
		bytes_per_pixel: 4,
		pixel_layout: PixelLayout::Bgra,
	};
	base.outgoing_source = Some(pyramid_buf);
	let cfg = ConfigBuilder::new(&base)
		.outgoing(PassBinding::OutgoingSource)
		.incoming(PassBinding::OutgoingSource)
		.dest(PassBinding::Inline(pyramid_buf))
		.dispatch_size(960, 540)
		.mip_levels(5)
		.build()
		.expect("builds");
	assert_eq!(cfg.outgoing_mip_levels, 5);
	assert_eq!(cfg.outgoing_data.unwrap() as usize, 0x3000);
	assert_eq!(cfg.dest_data as usize, 0x3000);
}

#[test]
fn host_capabilities_match_backend() {
	let base = make_test_base();
	let caps = base.capabilities();
	assert!(caps.supports(prgpu::effect::Capability::FrameExpansion));
	assert!(!caps.supports(prgpu::effect::Capability::SourceOutputMayAlias));
}
