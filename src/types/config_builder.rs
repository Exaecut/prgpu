//! Pass-level [`Configuration`] assembly.
//!
//! The graph executor builds one [`Configuration`] per pass via
//! [`ConfigBuilder`] using bindings drawn from the active
//! [`crate::effect::InvocationBase`]. Effect authors stop hand-mutating
//! `cfg.outgoing_data = ...; cfg.dest_pitch_px = ...;` etc. across every
//! kernel call.
//!
//! For now this is a thin builder over the existing [`Configuration`] ABI;
//! Phase 4-5 (the graph executor) is the primary consumer. Manual
//! construction stays available via `Configuration::cpu` /
//! `Configuration::effect` for code that hasn't migrated yet.

use crate::effect::{FrameBinding, InvocationBase, PixelLayout};
use crate::types::Configuration;

/// Reason a `ConfigBuilder::build` rejected a pass description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigBuildError {
	MissingDest,
	ZeroDispatchSize,
}

/// Either a borrowed [`FrameBinding`] from the [`InvocationBase`] or an
/// inline binding the pass owns (e.g. a freshly-allocated bloom pyramid).
#[derive(Debug, Clone, Copy)]
pub enum PassBinding {
	Output,
	MainSource,
	OutgoingSource,
	IncomingSource,
	Inline(FrameBinding),
	Null,
}

#[derive(Clone, Copy, Default)]
struct Size2D {
	width: u32,
	height: u32,
}

/// Assembles a [`Configuration`] for one pass.
///
/// Slot semantics follow the kernel-side binding contract enforced by every
/// Slang shader:
/// - **outgoing** = read-only main source (slot 0)
/// - **incoming** = read-only secondary source (slot 1)
/// - **dest** = RW destination (slot 2)
///
/// `source` / `input` / `target` are convenience aliases (source→outgoing,
/// input→incoming, target→dest) that match the plan's pass-DSL wording.
pub struct ConfigBuilder<'a> {
	base: &'a InvocationBase,
	outgoing: Option<PassBinding>,
	incoming: Option<PassBinding>,
	dest: Option<PassBinding>,
	dispatch: Option<Size2D>,
	outgoing_mip_levels: Option<u32>,
}

impl<'a> ConfigBuilder<'a> {
	pub fn new(base: &'a InvocationBase) -> Self {
		Self {
			base,
			outgoing: None,
			incoming: None,
			dest: None,
			dispatch: None,
			outgoing_mip_levels: None,
		}
	}

	pub fn outgoing(mut self, binding: PassBinding) -> Self {
		self.outgoing = Some(binding);
		self
	}

	pub fn incoming(mut self, binding: PassBinding) -> Self {
		self.incoming = Some(binding);
		self
	}

	pub fn dest(mut self, binding: PassBinding) -> Self {
		self.dest = Some(binding);
		self
	}

	pub fn source(self, binding: PassBinding) -> Self {
		self.outgoing(binding)
	}

	pub fn input(self, binding: PassBinding) -> Self {
		self.incoming(binding)
	}

	pub fn target(self, binding: PassBinding) -> Self {
		self.dest(binding)
	}

	pub fn dispatch_size(mut self, width: u32, height: u32) -> Self {
		self.dispatch = Some(Size2D { width, height });
		self
	}

	pub fn mip_levels(mut self, levels: u32) -> Self {
		self.outgoing_mip_levels = Some(levels);
		self
	}

	pub fn build(self) -> Result<Configuration, ConfigBuildError> {
		let dest_binding = match self.dest {
			Some(PassBinding::Null) | None => return Err(ConfigBuildError::MissingDest),
			Some(b) => self.resolve(b),
		};
		if dest_binding.data.is_null() {
			return Err(ConfigBuildError::MissingDest);
		}

		let dispatch = self
			.dispatch
			.unwrap_or(Size2D { width: dest_binding.width, height: dest_binding.height });
		if dispatch.width == 0 || dispatch.height == 0 {
			return Err(ConfigBuildError::ZeroDispatchSize);
		}

		let outgoing_binding = self.outgoing.map(|b| self.resolve(b)).unwrap_or_else(|| FrameBinding::null(self.base.bytes_per_pixel, self.base.pixel_layout));
		let incoming_binding = self.incoming.map(|b| self.resolve(b)).unwrap_or(outgoing_binding);

		let outgoing_data = if outgoing_binding.is_null() { None } else { Some(outgoing_binding.data) };
		let incoming_data = if incoming_binding.is_null() { None } else { Some(incoming_binding.data) };

		let outgoing_mip_levels = self.outgoing_mip_levels.unwrap_or(outgoing_binding.mip_levels);

		Ok(Configuration {
			device_handle: self.base.device_handle,
			context_handle: self.base.context_handle,
			command_queue_handle: self.base.command_queue_handle,
			outgoing_data,
			incoming_data,
			dest_data: dest_binding.data,
			outgoing_pitch_px: outgoing_binding.pitch_px,
			incoming_pitch_px: incoming_binding.pitch_px,
			dest_pitch_px: dest_binding.pitch_px,
			width: dispatch.width,
			height: dispatch.height,
			outgoing_width: outgoing_binding.width,
			outgoing_height: outgoing_binding.height,
			incoming_width: incoming_binding.width,
			incoming_height: incoming_binding.height,
			bytes_per_pixel: self.base.bytes_per_pixel,
			time: self.base.time,
			progress: self.base.progress,
			render_generation: self.base.render_generation,
			pixel_layout: self.base.pixel_layout.as_u32(),
			storage: self.base.storage,
			flip_y: self.base.flip_y,
			outgoing_mip_levels,
			canvas_width: self.base.output.width,
			canvas_height: self.base.output.height,
			layer_width: self.base.main_source.width,
			layer_height: self.base.main_source.height,
			ext_x: self.base.ext_x,
			ext_y: self.base.ext_y,
		})
	}

	fn resolve(&self, binding: PassBinding) -> FrameBinding {
		match binding {
			PassBinding::Output => self.base.output,
			PassBinding::MainSource => self.base.main_source,
			PassBinding::OutgoingSource => self.base.outgoing_source.unwrap_or(self.base.main_source),
			PassBinding::IncomingSource => self.base.incoming_source.unwrap_or(self.base.main_source),
			PassBinding::Inline(b) => b,
			PassBinding::Null => FrameBinding::null(self.base.bytes_per_pixel, PixelLayout::Bgra),
		}
	}
}
