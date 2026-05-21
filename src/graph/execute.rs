//! Graph execution.
//!
//! `execute_cpu` walks the declared graph against an [`InvocationBase`],
//! allocates resources from the CPU buffer pool, builds a fresh
//! `Configuration` per pass via `ConfigBuilder`, and routes each pass
//! through its type-erased dispatcher closure.
//!
//! GPU execution lands in Phase 5; the per-pass dispatcher closure already
//! knows how to call `Kernel<P>::dispatch_gpu` so this file's structure
//! will be reused unchanged.

use crate::cpu::buffer as cpu_buffer;
use crate::effect::{FrameBinding, InvocationBase, PixelLayout};
use crate::graph::builder::{GraphError, RenderGraph};
use crate::graph::context::{MipPyramidCtx, PassContext};
use crate::graph::pass::{MipChainPassDecl, MipDirection, PassDecl, SinglePassDecl, Slot};
use crate::graph::resource::{MipPyramidDesc, ResourceId};
use crate::types::{ConfigBuilder, Configuration, ImageBuffer, PassBinding};

struct AllocatedResource {
	#[allow(dead_code)]
	desc: MipPyramidDesc,
	buffer: ImageBuffer,
}

impl AllocatedResource {
	fn binding_for(&self, lod: Option<u32>, base: &InvocationBase) -> FrameBinding {
		// Mip chains use a tightly-packed buffer; the kernel resolves per-lod
		// offsets from `frame.outDesc.mipOffsetBytes[]`. The pass binding always
		// targets the base allocation with `mip_levels` set so `make_outgoing_desc`
		// can populate the descriptor for every lod.
		let _ = lod;
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

/// Execute a CPU graph end-to-end against `base`. Resources are allocated
/// from the CPU buffer pool keyed by `MipPyramidDesc::tag`. Errors short-
/// circuit: a failure mid-graph aborts the whole render to surface as an
/// effect-level error rather than producing a partial output.
pub fn execute_cpu<F>(graph: &RenderGraph<F>, frame_data: &F, base: &InvocationBase) -> Result<(), GraphError>
where
	F: Copy + Send + Sync + 'static,
{
	let mut resources: Vec<AllocatedResource> = Vec::with_capacity(graph.resources.len());
	for (idx, decl) in graph.resources.iter().enumerate() {
		let ctx = MipPyramidCtx::new(frame_data, base);
		let desc = (decl.desc_fn)(&ctx);
		let buffer = cpu_buffer::get_or_create_with_mips(desc.base_width, desc.base_height, base.bytes_per_pixel, desc.levels.max(1), desc.tag);
		if buffer.buf.raw.is_null() {
			return Err(GraphError::ResourceAllocFailed { name: decl.name });
		}
		let _ = idx;
		resources.push(AllocatedResource { desc, buffer });
	}

	for pass in &graph.passes {
		match pass {
			PassDecl::Single(p) => execute_single_cpu(p, frame_data, base, &resources)?,
			PassDecl::MipChain(p) => execute_mip_chain_cpu(p, frame_data, base, &resources)?,
		}
	}

	Ok(())
}

fn execute_single_cpu<F>(pass: &SinglePassDecl<F>, frame_data: &F, base: &InvocationBase, resources: &[AllocatedResource]) -> Result<(), GraphError>
where
	F: Copy + Send + Sync + 'static,
{
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

	let ctx = PassContext::new(frame_data, base);
	(pass.dispatcher)(&ctx, &config).map_err(|m| GraphError::KernelDispatch { pass: pass.name, message: m })
}

fn execute_mip_chain_cpu<F>(pass: &MipChainPassDecl<F>, frame_data: &F, base: &InvocationBase, resources: &[AllocatedResource]) -> Result<(), GraphError>
where
	F: Copy + Send + Sync + 'static,
{
	let res = resources.get(pass.resource.0 as usize).ok_or(GraphError::UnknownResource(pass.name))?;
	let levels = res.desc.levels.max(1);
	let binding = res.binding_for(None, base);

	let ctx = PassContext::new(frame_data, base);

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
			.outgoing(PassBinding::Inline(binding))
			.incoming(PassBinding::Inline(binding))
			.dest(PassBinding::Inline(binding))
			.dispatch_size(dst_w, dst_h)
			.mip_levels(levels)
			.build()
			.map_err(|e| GraphError::ConfigBuild { pass: pass.name, kind: e })?;

		(pass.dispatcher)(level, &ctx, &config).map_err(|m| GraphError::KernelDispatch { pass: pass.name, message: m })?;
	}

	Ok(())
}

fn resolve_slot(slot: Slot, base: &InvocationBase, resources: &[AllocatedResource], pass_name: Option<&'static str>) -> Result<FrameBinding, GraphError> {
	match slot {
		Slot::MainSource => Ok(base.main_source),
		Slot::Output => Ok(base.output),
		Slot::Inline(b) => Ok(b),
		Slot::ResourceWhole(id) => {
			let r = resources.get(id.0 as usize).ok_or_else(|| GraphError::UnknownResource(pass_name.unwrap_or("?")))?;
			Ok(r.binding_for(None, base))
		}
		Slot::ResourceMip(id, lod) => {
			let r = resources.get(id.0 as usize).ok_or_else(|| GraphError::UnknownResource(pass_name.unwrap_or("?")))?;
			let max = r.desc.levels;
			if lod >= max {
				return Err(GraphError::BadMipLevel {
					pass: pass_name.unwrap_or("?"),
					level: lod,
					max,
				});
			}
			let mut binding = r.binding_for(Some(lod), base);
			binding.width = (r.buffer.width >> lod).max(1);
			binding.height = (r.buffer.height >> lod).max(1);
			Ok(binding)
		}
	}
}

#[allow(unused)]
fn unused_silencer(_: PixelLayout, _: ResourceId, _: Configuration) {}
