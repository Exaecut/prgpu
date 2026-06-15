//! Graph execution.
//!
//! [`execute`] walks the declared graph against an [`InvocationBase`],
//! allocates resources from the CPU or GPU buffer pool (picked by
//! `base.backend`), builds a fresh `Configuration` per pass via
//! `ConfigBuilder`, and routes each pass through its type-erased dispatcher
//! closure. The dispatcher closure itself decides between
//! `Kernel<P>::dispatch_cpu_direct` and `Kernel<P>::dispatch_gpu` based on
//! the active backend.

use crate::cpu::buffer as cpu_buffer;
use crate::effect::{Capability, Ctx, FrameBinding, InvocationBase};
use crate::graph::pass::{MipChainPassDecl, MipDirection, PassDecl, SinglePassDecl, Slot};
use crate::graph::resource::{MipPyramidDesc, ResourceId};
use crate::graph::source::{SourcePolicy, AUTO_SOURCE_SNAPSHOT_TAG};
use crate::params::ParamsSpec;
use crate::pipeline::mip;
use crate::types::{Backend, ConfigBuildError, ConfigBuilder, Configuration, DeviceHandleInit, ImageBuffer, PassBinding};

use crate::graph::builder::Graph;

struct AllocatedResource {
	desc: MipPyramidDesc,
	buffer: ImageBuffer,
}

impl AllocatedResource {
	fn binding_for(&self, _lod: u32, base: &InvocationBase) -> FrameBinding {
		FrameBinding {
			data: self.buffer.buf.raw,
			pitch_px: self.buffer.pitch_px as i32,
			width: self.buffer.width,
			height: self.buffer.height,
			mip_levels: self.desc.levels,
			bytes_per_pixel: self.buffer.bytes_per_pixel,
			pixel_layout: base.pixel_layout,
		}
	}
}

#[derive(Debug)]
pub enum GraphError {
	ResourceAllocFailed { name: &'static str },
	UnknownResource(&'static str),
	BadMipLevel { pass: &'static str, level: u32, max: u32 },
	ConfigBuild { pass: &'static str, kind: ConfigBuildError },
	KernelDispatch { pass: &'static str, message: &'static str },
}

/// Execute a graph end-to-end against `ctx` and `base`. Resources are
/// allocated from the CPU buffer pool when `base.backend == Cpu`, the GPU
/// buffer pool otherwise. Errors short-circuit: a failure mid-graph aborts
/// the whole render to surface as an effect-level error rather than producing
/// a partial output.
///
/// `Source` into a private buffer when the host signals
/// `Capability::SourceOutputMayAlias` and a pass reads `Source` while
/// writing `Output`; `SnapshotIfAliased { tag }` does the same with a caller
/// tag; `AlwaysSnapshot` skips the capability check; `Direct` never copies.
/// When a snapshot is taken the executor rebinds `base.source` to it for
/// the rest of the graph.
pub fn execute<P: ParamsSpec>(graph: &Graph<P>, ctx: &Ctx<P>, base: &InvocationBase) -> Result<(), GraphError> {
	let mut local_base = clone_base(base);
	let auto_snapshot_needed = graph_samples_source_into_output(graph);
	let _snapshot_buf = apply_source_policy(&mut local_base, graph.source_policy, auto_snapshot_needed)?;

	let mut resources: Vec<AllocatedResource> = Vec::with_capacity(graph.resources.len());
	for decl in &graph.resources {
		let desc = (decl.desc_fn)(ctx);
		let buffer = match local_base.backend {
			Backend::Cpu => cpu_buffer::get_or_create_with_mips(desc.base_width, desc.base_height, local_base.bytes_per_pixel, desc.levels.max(1), desc.tag),
			Backend::Cuda | Backend::Metal => unsafe { crate::gpu::buffer::get_or_create_with_mips(DeviceHandleInit::FromPtr(local_base.device_handle), desc.base_width, desc.base_height, local_base.bytes_per_pixel, desc.levels.max(1), desc.tag) },
		};
		if buffer.buf.raw.is_null() {
			return Err(GraphError::ResourceAllocFailed { name: decl.name });
		}

		if desc.populate_from_source && !local_base.source.is_null() {
			let mut tmp_cfg = Configuration {
				device_handle: local_base.device_handle,
				context_handle: local_base.context_handle,
				command_queue_handle: local_base.command_queue_handle,
				outgoing_data: Some(local_base.source.data),
				incoming_data: Some(local_base.source.data),
				dest_data: buffer.buf.raw,
				outgoing_pitch_px: local_base.source.pitch_px,
				incoming_pitch_px: local_base.source.pitch_px,
				dest_pitch_px: buffer.pitch_px as i32,
				width: desc.base_width,
				height: desc.base_height,
				outgoing_width: local_base.source.width,
				outgoing_height: local_base.source.height,
				incoming_width: local_base.source.width,
				incoming_height: local_base.source.height,
				bytes_per_pixel: local_base.bytes_per_pixel,
				time: local_base.time,
				progress: local_base.progress,
				render_generation: local_base.render_generation,
				pixel_layout: local_base.pixel_layout.as_u32(),
				storage: local_base.storage,
				flip_y: local_base.flip_y,
				outgoing_mip_levels: desc.levels,
				canvas_width: local_base.output.width,
				canvas_height: local_base.output.height,
				layer_width: local_base.source.width,
				layer_height: local_base.source.height,
				ext_x: local_base.ext_x,
				ext_y: local_base.ext_y,
			};
			unsafe {
				mip::prepare_mip_source(&mut tmp_cfg, desc.tag).map_err(|m| GraphError::KernelDispatch { pass: "prepare_mip_resource", message: m })?;
				mip::generate_mips(&tmp_cfg).map_err(|m| GraphError::KernelDispatch { pass: "generate_mip_resource", message: m })?;
			}
		}

		resources.push(AllocatedResource { desc, buffer });
	}

	for pass in &graph.passes {
		match pass {
			PassDecl::Single(p) => {
				let enabled = p.enabled_when.as_ref().map(|f| f(ctx)).unwrap_or(true);
				if enabled {
					execute_single(p, ctx, &local_base, &resources)?;
				}
			}
			PassDecl::MipChain(p) => {
				let enabled = p.enabled_when.as_ref().map(|f| f(ctx)).unwrap_or(true);
				if enabled {
					execute_mip_chain(p, ctx, &local_base, &resources)?;
				}
			}
		}
	}

	Ok(())
}

fn clone_base(base: &InvocationBase) -> InvocationBase {
	InvocationBase {
		host: base.host,
		backend: base.backend,
		render_kind: base.render_kind,
		device_handle: base.device_handle,
		context_handle: base.context_handle,
		command_queue_handle: base.command_queue_handle,
		bytes_per_pixel: base.bytes_per_pixel,
		pixel_layout: base.pixel_layout,
		storage: base.storage,
		flip_y: base.flip_y,
		time: base.time,
		progress: base.progress,
		render_generation: base.render_generation,
		ext_x: base.ext_x,
		ext_y: base.ext_y,
		source: base.source,
		layers: base.layers,
		output: base.output,
	}
}

fn graph_samples_source_into_output<P: ParamsSpec>(graph: &Graph<P>) -> bool {
	graph.passes.iter().any(|pass| match pass {
		PassDecl::Single(p) => {
			let reads_source = matches!(p.source, Slot::Source) || matches!(p.input, Some(Slot::Source));
			reads_source && matches!(p.target, Slot::Output)
		}
		PassDecl::MipChain(_) => false,
	})
}

fn apply_source_policy(base: &mut InvocationBase, policy: SourcePolicy, auto_snapshot_needed: bool) -> Result<Option<ImageBuffer>, GraphError> {
	let tag = match policy {
		SourcePolicy::Direct => return Ok(None),
		SourcePolicy::Auto => {
			if !auto_snapshot_needed || !base.capabilities().supports(Capability::SourceOutputMayAlias) {
				return Ok(None);
			}
			AUTO_SOURCE_SNAPSHOT_TAG
		}
		SourcePolicy::SnapshotIfAliased { tag } => {
			if !base.capabilities().supports(Capability::SourceOutputMayAlias) {
				return Ok(None);
			}
			tag
		}
		SourcePolicy::AlwaysSnapshot { tag } => tag,
	};

	if base.source.is_null() {
		return Ok(None);
	}

	let mut tmp_cfg = Configuration {
		device_handle: base.device_handle,
		context_handle: base.context_handle,
		command_queue_handle: base.command_queue_handle,
		outgoing_data: Some(base.source.data),
		incoming_data: Some(base.source.data),
		dest_data: base.output.data,
		outgoing_pitch_px: base.source.pitch_px,
		incoming_pitch_px: base.source.pitch_px,
		dest_pitch_px: base.output.pitch_px,
		width: base.source.width,
		height: base.source.height,
		outgoing_width: base.source.width,
		outgoing_height: base.source.height,
		incoming_width: base.source.width,
		incoming_height: base.source.height,
		bytes_per_pixel: base.bytes_per_pixel,
		time: base.time,
		progress: base.progress,
		render_generation: base.render_generation,
		pixel_layout: base.pixel_layout.as_u32(),
		storage: base.storage,
		flip_y: base.flip_y,
		outgoing_mip_levels: 0,
		canvas_width: base.output.width,
		canvas_height: base.output.height,
		layer_width: base.source.width,
		layer_height: base.source.height,
		ext_x: base.ext_x,
		ext_y: base.ext_y,
	};

	let snapshot = unsafe { mip::prepare_source_copy(&mut tmp_cfg, tag) }.map_err(|m| GraphError::KernelDispatch { pass: "source_snapshot", message: m })?;

	base.source = FrameBinding {
		data: snapshot.buf.raw,
		pitch_px: snapshot.pitch_px as i32,
		width: snapshot.width,
		height: snapshot.height,
		mip_levels: 0,
		bytes_per_pixel: snapshot.bytes_per_pixel,
		pixel_layout: base.pixel_layout,
	};

	Ok(Some(snapshot))
}

fn execute_single<P: ParamsSpec>(pass: &SinglePassDecl<P>, ctx: &Ctx<P>, base: &InvocationBase, resources: &[AllocatedResource]) -> Result<(), GraphError> {
	let target_binding = resolve_slot(pass.target, base, resources, Some(pass.name))?;
	let source_binding = resolve_slot(pass.source, base, resources, Some(pass.name))?;
	let input_binding = match pass.input {
		Some(slot) => Some(resolve_slot(slot, base, resources, Some(pass.name))?),
		None => None,
	};

	let mut builder = ConfigBuilder::new(base).source(PassBinding::Inline(source_binding)).target(PassBinding::Inline(target_binding)).dispatch_size(target_binding.width, target_binding.height);
	if let Some(i) = input_binding {
		builder = builder.input(PassBinding::Inline(i));
	} else {
		builder = builder.input(PassBinding::Inline(source_binding));
	}
	if source_binding.mip_levels > 1 {
		builder = builder.mip_levels(source_binding.mip_levels);
	}

	let config = builder.build().map_err(|e| GraphError::ConfigBuild { pass: pass.name, kind: e })?;

	(pass.dispatcher)(ctx, &config).map_err(|m| GraphError::KernelDispatch { pass: pass.name, message: m })
}

fn execute_mip_chain<P: ParamsSpec>(pass: &MipChainPassDecl<P>, ctx: &Ctx<P>, base: &InvocationBase, resources: &[AllocatedResource]) -> Result<(), GraphError> {
	let res = resources.get(pass.resource.0 as usize).ok_or(GraphError::UnknownResource(pass.name))?;
	let levels = res.desc.levels.max(1);
	let binding = res.binding_for(0, base);

	let level_iter: Box<dyn Iterator<Item = u32>> = match pass.direction {
		MipDirection::Down => Box::new(0..levels.saturating_sub(1)),
		MipDirection::Up => Box::new((0..levels.saturating_sub(1)).rev()),
	};

	for level in level_iter {
		let dst_lod = match pass.direction {
			MipDirection::Down => level + 1,
			MipDirection::Up => level,
		};
		let dst_w = (binding.width >> dst_lod).max(1);
		let dst_h = (binding.height >> dst_lod).max(1);

		let config = ConfigBuilder::new(base)
			.source(PassBinding::Inline(binding))
			.input(PassBinding::Inline(binding))
			.target(PassBinding::Inline(binding))
			.dispatch_size(dst_w, dst_h)
			.mip_levels(levels)
			.build()
			.map_err(|e| GraphError::ConfigBuild { pass: pass.name, kind: e })?;

		(pass.dispatcher)(level, ctx, &config).map_err(|m| GraphError::KernelDispatch { pass: pass.name, message: m })?;
	}

	Ok(())
}

fn resolve_slot(slot: Slot, base: &InvocationBase, resources: &[AllocatedResource], pass_name: Option<&'static str>) -> Result<FrameBinding, GraphError> {
	match slot {
		Slot::Source => Ok(base.source),
		Slot::Output => Ok(base.output),
		Slot::Inline(b) => Ok(b),
		Slot::Layer(idx) => {
			// Source of truth is the actual checkout; a missing/null aux layer
			// falls back to the main source so a pass never reads a dangling
			// pointer. Pipelines that must branch on presence gate themselves
			// with `Ctx::layer_present` (same flags as `base.layer_presence`).
			let binding = base.layers.get(idx as usize).copied().flatten();
			match binding {
				Some(b) if !b.is_null() => Ok(b),
				_ => Ok(base.source),
			}
		}
		Slot::Mip(handle, level) => {
			let r = resources.get(handle.id.0 as usize).ok_or_else(|| GraphError::UnknownResource(pass_name.unwrap_or("?")))?;
			let max = r.desc.levels;
			if level >= max {
				return Err(GraphError::BadMipLevel {
					pass: pass_name.unwrap_or("?"),
					level,
					max,
				});
			}
			let mut binding = r.binding_for(level, base);
			binding.width = (r.buffer.width >> level).max(1);
			binding.height = (r.buffer.height >> level).max(1);
			Ok(binding)
		}
	}
}
