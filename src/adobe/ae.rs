//! After Effects PF adapter: drives every selector through the
//! `Effect` trait so per-effect crates stop hand-writing `handle_command`.
//!
//! Selector mapping:
//!
//! | AE selector             | Effect method                                      |
//! |-------------------------|----------------------------------------------------|
//! | `Cmd_GlobalSetup`       | `Effect::descriptor` + `LicenseGate::initialize`   |
//! | `Cmd_About`             | descriptor `about_text`                            |
//! | `Cmd_ParamsSetup`       | `P::register` + `Effect::extra_params`             |
//! | `Cmd_UpdateParamsUi`    | `Effect::ui` + visibility from snapshot            |
//! | `Cmd_UserChangedParam`  | `P::buttons` callback                              |
//! | `Cmd_FrameSetup`        | `Effect::expansion` + snapshot                     |
//! | `Cmd_FrameSetdown`      | drop frame state                                   |
//! | `Cmd_Render`            | graph execution (CPU)                              |
//! | `Cmd_SmartPreRender`    | expansion + snapshot                               |
//! | `Cmd_SmartRender`       | graph execution (CPU)                              |
//! | `Cmd_SmartRenderGpu`    | graph execution (GPU)                              |
//! | `Cmd_GpuDeviceSetup`    | declares supported GPU framework                   |
//!
//! A typical handwritten `lib.rs` (`handle_command` + per-host parameter
//! visibility + manual licence checks + per-pass `Configuration` mutation)
//! collapses into a single `impl Effect` plus a thin `AdobePluginGlobal`
//! trampoline.

use std::ffi::c_void;
use std::sync::OnceLock;

use after_effects::aegp;
use after_effects::{self as ae, Command, GpuFramework, InData, OutData, Parameters};

use crate::effect::descriptor::install_descriptor_pixel_formats;
use crate::effect::host::{Host, HostCapabilities, RenderKind};
use crate::effect::{
	Ctx, Effect, EffectDescriptor, FrameBinding, Geometry, InvocationBase, LicenseGate,
	PixelLayout, Timing, Ui,
};
use crate::graph::{Graph, execute::execute as run_graph};
use crate::params::{ParamsSpec, SnapshotGeom};
use crate::types::Backend;

/// Stored per-frame via AE's `FrameData` mechanism. Replaces the old
/// `FrameData` type param: all per-frame context is baked into the snapshot.
struct FrameState<P: ParamsSpec> {
	snapshot: P::Snapshot,
	geom: SnapshotGeom,
	time_seconds: f32,
}

impl<P: ParamsSpec> FrameState<P> {
	fn ctx(&self, host: Host, backend: Backend, frame_index: u32, progress: f32, debug_view: bool) -> Ctx<'_, P> {
		let timing = Timing {
			frame_index,
			time_seconds: self.time_seconds,
			progress,
		};
		let caps = HostCapabilities::new(host, backend);
		let geom = Geometry {
			layer_w: self.geom.layer_w,
			layer_h: self.geom.layer_h,
			output_w: self.geom.output_w,
			output_h: self.geom.output_h,
			ext_x: self.geom.ext_x,
			ext_y: self.geom.ext_y,
		};
		Ctx::new(&self.snapshot, geom, timing, caps, debug_view)
	}
}

/// Canonical effect time in seconds. In Premiere this is the sequence/timeline
/// time (`PF_UtilitySuite::GetSequenceTime`), matching the GPU path's
/// `RenderParams::sequence_time`. In After Effects (no sequence) it falls back
/// to layer-local `current_time / time_scale`.
fn canonical_time_seconds(in_data: &InData) -> f32 {
	if in_data.is_premiere() {
		if let Ok(suite) = ae::pf::suites::Utility::new() {
			if let Ok(ticks) = suite.sequence_time(in_data.effect_ref()) {
				return crate::adobe::ticks_to_seconds(ticks);
			}
		}
	}
	if in_data.time_scale() != 0 {
		in_data.current_time() as f32 / in_data.time_scale() as f32
	} else {
		0.0
	}
}

fn backend_from_cfg() -> Backend {
	#[cfg(gpu_backend = "cuda")]
	{
		Backend::Cuda
	}
	#[cfg(gpu_backend = "metal")]
	{
		Backend::Metal
	}
	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	{
		Backend::Cpu
	}
}

fn host_from_in_data(in_data: &InData) -> Host {
	if in_data.is_premiere() {
		Host::Premiere
	} else {
		Host::AfterEffects
	}
}

/// AE PF adapter. Implements [`AdobePluginGlobal`] over the [`Effect`] trait
/// so `ae::define_effect!(Plugin, (), Params)` can register a plugin whose
/// only declarative content lives in `impl Effect for MyEffect`.
pub struct EffectAdapter<E: Effect, L: LicenseGate> {
	license: L,
	graph: OnceLock<Graph<E::Params>>,
	descriptor: OnceLock<EffectDescriptor>,
	plugin_id: aegp::PluginId,
	ui_rules: OnceLock<Vec<(E::Params, Box<dyn Fn(&Ctx<E::Params>) -> bool + Send + Sync + 'static>)>>,
}

impl<E: Effect, L: LicenseGate> Default for EffectAdapter<E, L> {
	fn default() -> Self {
		Self {
			license: L::default(),
			graph: OnceLock::new(),
			descriptor: OnceLock::new(),
			plugin_id: aegp::PluginId::default(),
			ui_rules: OnceLock::new(),
		}
	}
}

impl<E: Effect, L: LicenseGate> EffectAdapter<E, L> {
	fn descriptor(&self) -> &EffectDescriptor {
		self.descriptor
			.get_or_init(|| E::descriptor(EffectDescriptor::new("")))
	}

	fn graph(&self) -> &Graph<E::Params> {
		self.graph.get_or_init(|| {
			let mut g = Graph::new();
			E::pipeline(&mut g);
			g
		})
	}

	/// License gate consulted before every render selector. In debug builds a
	/// closed gate logs the failing state label so a blank render is traceable
	/// in the AE console; release inlines to the bare `is_valid()`.
	#[inline]
	fn license_valid(&self) -> bool {
		let ok = self.license.is_valid();
		#[cfg(debug_assertions)]
		if !ok {
			log::warn!(
				"license: gate closed, render skipped; state=[{}]",
				self.license.debug_label().unwrap_or_default()
			);
		}
		ok
	}

	fn snapshot_and_ctx(
		params: &Parameters<E::Params>,
		geom: &SnapshotGeom,
		host: Host,
		backend: Backend,
		time_seconds: f32,
	) -> Result<FrameState<E::Params>, ae::Error> {
		let snapshot = E::Params::snapshot_cpu(params, geom)?;
		Ok(FrameState {
			snapshot,
			geom: *geom,
			time_seconds,
		})
	}

	fn ensure_ui_rules(&self) {
		if self.ui_rules.get().is_some() {
			return;
		}
		let mut ui = Ui::new();
		E::ui(&mut ui);
		let _ = self.ui_rules.set(ui.rules);
	}

	fn apply_visibility(
		&self,
		params: &mut Parameters<E::Params>,
		in_data: &InData,
		out_data: &mut OutData,
	) -> Result<(), ae::Error> {
		let host = host_from_in_data(in_data);
		let backend = backend_from_cfg();
		let time_seconds = canonical_time_seconds(in_data);

		let layer_w = 1u32;
		let layer_h = 1u32;
		let geom = SnapshotGeom {
			layer_w,
			layer_h,
			output_w: layer_w,
			output_h: layer_h,
			ext_x: 0,
			ext_y: 0,
		};
		let state = Self::snapshot_and_ctx(params, &geom, host, backend, time_seconds)?;
		let ctx = state.ctx(host, backend, 0, 0.0, false);

		let rules = self.ui_rules.get();
		let Some(rules) = rules else {
			return Ok(());
		};

		let mut visible_map: Vec<(E::Params, bool)> = Vec::with_capacity(rules.len());
		for (param_id, pred) in rules {
			let visible = pred(&ctx);
			visible_map.push((*param_id, visible));
			if let Ok(mut p) = params.get_mut(*param_id) {
				p.set_ui_flag(ae::ParamUIFlags::INVISIBLE, !visible);
				let _ = p.update_param_ui();
			}
		}

		if in_data.is_after_effects() && self.plugin_id != aegp::PluginId::default() {
			let effect = in_data.effect();
			let plugin_id = self.plugin_id;
			if let Ok(aegp_plugin) = effect.aegp_effect(plugin_id) {
				for (id, visible) in &visible_map {
					if let Some(idx) = params.index(*id) {
						if let Ok(stream) = aegp_plugin.new_stream_by_index(plugin_id, idx as _) {
							let _ = stream.set_dynamic_stream_flag(
								aegp::DynamicStreamFlags::Hidden,
								false,
								!*visible,
							);
						}
					}
				}
			}
		}

		out_data.set_out_flag(ae::OutFlags::RefreshUi, true);
		Ok(())
	}

	fn build_invocation_cpu(
		in_data: &InData,
		in_layer: &ae::Layer,
		out_layer: &mut ae::Layer,
	) -> Result<InvocationBase, ae::Error> {
		let bpp = crate::cpu::render::compute_bpp(in_data, out_layer)?;
		let pixel_layout =
			PixelLayout::from_u32(crate::cpu::render::pixel_layout_from_format(in_data, in_layer));

		let in_w = in_layer.width() as u32;
		let in_h = in_layer.height() as u32;
		let out_w = out_layer.width() as u32;
		let out_h = out_layer.height() as u32;

		let in_ptr = in_layer.buffer().as_ptr() as *mut c_void;
		let out_ptr = out_layer.buffer_mut().as_mut_ptr() as *mut c_void;
		let src_pitch = in_layer.buffer_stride() as i32 / bpp as i32;
		let dest_pitch = out_layer.buffer_stride() as i32 / bpp as i32;

		let host = host_from_in_data(in_data);
		let render_kind = if in_data.is_premiere() {
			RenderKind::PremiereGpuEffect
		} else {
			RenderKind::AeSmartRenderCpu
		};

		let main = FrameBinding {
			data: in_ptr,
			pitch_px: src_pitch,
			width: in_w,
			height: in_h,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};
		let output = FrameBinding {
			data: out_ptr,
			pitch_px: dest_pitch,
			width: out_w,
			height: out_h,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};

		Ok(InvocationBase {
			host,
			backend: Backend::Cpu,
			render_kind,
			device_handle: std::ptr::null_mut(),
			context_handle: None,
			command_queue_handle: std::ptr::null_mut(),
			bytes_per_pixel: bpp,
			pixel_layout,
			storage: crate::types::storage_from_bpp(bpp),
			flip_y: in_data.is_premiere() as u32,
			time: canonical_time_seconds(in_data),
			progress: 0.0,
			render_generation: 0,
			ext_x: ((out_w as i32 - in_w as i32) / 2).max(0),
			ext_y: ((out_h as i32 - in_h as i32) / 2).max(0),
			source: main,
			layers: [None; crate::effect::invocation::MAX_AUX_LAYERS],
			output,
		})
	}

	fn build_invocation_gpu(
		in_data: &InData,
		in_layer: &mut ae::Layer,
		out_layer: &mut ae::Layer,
		extra: &ae::pf::SmartRenderExtra,
	) -> Result<InvocationBase, ae::Error> {
		let gpu_suite = ae::pf::suites::GPUDevice::new()?;
		let device_index = extra.device_index();
		let info = gpu_suite.device_info(in_data.effect_ref(), device_index)?;

		let src_mem = gpu_suite.gpu_world_data(in_data.effect_ref(), &mut *in_layer)?;
		let dst_mem = gpu_suite.gpu_world_data(in_data.effect_ref(), &mut *out_layer)?;

		let bpp = match out_layer.pixel_format() {
			Ok(ae::pf::PixelFormat::GpuBgra128) | Ok(ae::pf::PixelFormat::Argb128) => 16u32,
			Ok(ae::pf::PixelFormat::Argb64) => 8u32,
			_ => crate::cpu::render::compute_bpp(in_data, out_layer)?,
		};
		let pixel_layout =
			PixelLayout::from_u32(crate::cpu::render::pixel_layout_from_format(in_data, in_layer));

		let in_w = in_layer.width() as u32;
		let in_h = in_layer.height() as u32;
		let out_w = out_layer.width() as u32;
		let out_h = out_layer.height() as u32;
		let src_pitch = in_layer.buffer_stride() as i32 / bpp as i32;
		let dest_pitch = out_layer.buffer_stride() as i32 / bpp as i32;

		#[cfg(gpu_backend = "metal")]
		let device_ptr = info.devicePV;
		#[cfg(gpu_backend = "cuda")]
		let device_ptr = info.contextPV;
		#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
		let device_ptr = std::ptr::null_mut();

		let backend = match extra.what_gpu() {
			GpuFramework::Cuda => Backend::Cuda,
			GpuFramework::Metal => Backend::Metal,
			_ => Backend::Cpu,
		};

		let frame_index = {
			let step = in_data.time_step().max(1);
			(in_data.current_time() / step).max(0) as u32
		};

		let main = FrameBinding {
			data: src_mem,
			pitch_px: src_pitch,
			width: in_w,
			height: in_h,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};
		let output = FrameBinding {
			data: dst_mem,
			pitch_px: dest_pitch,
			width: out_w,
			height: out_h,
			mip_levels: 0,
			bytes_per_pixel: bpp,
			pixel_layout,
		};

		Ok(InvocationBase {
			host: host_from_in_data(in_data),
			backend,
			render_kind: RenderKind::AeSmartRenderGpu,
			device_handle: device_ptr as *mut c_void,
			context_handle: if info.contextPV.is_null() {
				None
			} else {
				Some(info.contextPV as *mut c_void)
			},
			command_queue_handle: info.command_queuePV as *mut c_void,
			bytes_per_pixel: bpp,
			pixel_layout,
			storage: crate::types::storage_from_bpp(bpp),
			flip_y: 0,
			time: canonical_time_seconds(in_data),
			progress: 0.0,
			render_generation: frame_index as u64,
			ext_x: ((out_w as i32 - in_w as i32) / 2).max(0),
			ext_y: ((out_h as i32 - in_h as i32) / 2).max(0),
			source: main,
			layers: [None; crate::effect::invocation::MAX_AUX_LAYERS],
			output,
		})
	}

	/// SmartPreRender checkout id for the k-th `#[layer]` param. Id 0 is the
	/// main input; aux layers start at 1 and are re-used at SmartRender.
	#[inline]
	fn aux_checkout_id(k: usize) -> i32 {
		1 + k as i32
	}

	/// Check out each declared layer param's pixels (already requested in
	/// SmartPreRender) and build a `FrameBinding` per slot. Returns the kept
	/// `Layer` worlds (their buffers must outlive the graph run) alongside the
	/// per-slot bindings. `gpu_suite` is `Some` on the GPU path (device-pointer
	/// extraction), `None` on the CPU path (host raster pointer).
	fn checkout_aux_layers(
		in_data: &InData,
		cb: &ae::pf::SmartRenderCallbacks,
		bpp: u32,
		pixel_layout: PixelLayout,
		gpu_suite: Option<&ae::pf::suites::GPUDevice>,
	) -> (Vec<(usize, ae::Layer)>, [Option<FrameBinding>; crate::effect::invocation::MAX_AUX_LAYERS]) {
		let mut bindings = [None; crate::effect::invocation::MAX_AUX_LAYERS];
		let mut worlds: Vec<(usize, ae::Layer)> = Vec::new();
		for k in 0..E::Params::LAYER_PARAMS.len().min(crate::effect::invocation::MAX_AUX_LAYERS) {
			let Ok(Some(mut layer)) = cb.checkout_layer_pixels(Self::aux_checkout_id(k) as u32) else {
				continue;
			};
			let w = layer.width() as u32;
			let h = layer.height() as u32;
			if w == 0 || h == 0 {
				continue;
			}
			let data = match gpu_suite {
				Some(suite) => match suite.gpu_world_data(in_data.effect_ref(), &mut layer) {
					Ok(ptr) => ptr,
					Err(_) => continue,
				},
				None => layer.buffer().as_ptr() as *mut c_void,
			};
			if data.is_null() {
				continue;
			}
			let pitch_px = if bpp > 0 { layer.buffer_stride() as i32 / bpp as i32 } else { 0 };
			bindings[k] = Some(FrameBinding {
				data,
				pitch_px,
				width: w,
				height: h,
				mip_levels: 0,
				bytes_per_pixel: bpp,
				pixel_layout,
			});
			worlds.push((k, layer));
		}
		(worlds, bindings)
	}

	fn run_graph(&self, ctx: &Ctx<E::Params>, base: &InvocationBase) -> Result<(), ae::Error> {
		use crate::gpu::frame_scope;
		let scope_desc = crate::types::FrameScopeDesc::from_invocation(base);
		const MAX_FRAME_ATTEMPTS: u32 = 2;
		for attempt in 1..=MAX_FRAME_ATTEMPTS {
			frame_scope::begin(&scope_desc);
			let result = run_graph(self.graph(), ctx, base);
			let sync = frame_scope::end(&scope_desc);
			result.map_err(|_| ae::Error::Generic)?;
			match sync {
				Ok(()) => return Ok(()),
				Err(e) if e == frame_scope::ERR_WATCHDOG && attempt < MAX_FRAME_ATTEMPTS => {
					log::warn!(
						"[prgpu] frame hit GPU watchdog (attempt {attempt}/{MAX_FRAME_ATTEMPTS}) — cooling down 50ms and retrying"
					);
					std::thread::sleep(std::time::Duration::from_millis(50));
				}
				Err(_) => return Err(ae::Error::Generic),
			}
		}
		Err(ae::Error::Generic)
	}
}

impl<E: Effect, L: LicenseGate> EffectAdapter<E, L> {
	/// AE PF `params_setup` selector. Call from the user's
	/// `AdobePluginGlobal::params_setup` impl (generated by
	/// `define_effect!`).
	pub fn params_setup(
		&self,
		params: &mut Parameters<E::Params>,
		in_data: InData,
		_out_data: OutData,
	) -> Result<(), ae::Error> {
		if let Some(label) = self.descriptor().options_button {
			let _ = in_data.effect().set_options_button_name(label);
		}
		E::Params::register(params)?;
		E::extra_params(params)?;
		Ok(())
	}

	/// AE PF `handle_command` selector. Call from the user's
	/// `AdobePluginGlobal::handle_command` impl.
	pub fn handle_command(
		&mut self,
		command: Command,
		in_data: InData,
		mut out_data: OutData,
		params: &mut Parameters<E::Params>,
	) -> Result<(), ae::Error> {
		match command {
			Command::GlobalSetup => {
				#[cfg(target_os = "windows")]
				let _ = ae::log::set_logger(&ae::win_dbg_logger::DEBUGGER_LOGGER);
				#[cfg(target_os = "macos")]
				let _ = ae::oslog::OsLogger::new(env!("CARGO_PKG_NAME")).init();
				ae::log::set_max_level(ae::log::LevelFilter::Info);

				install_descriptor_pixel_formats(&in_data, self.descriptor())?;

				if let Some(label) = self.descriptor().options_button {
					let _ = in_data.effect().set_options_button_name(label);
				}
				if let Ok(suite) = aegp::suites::Utility::new() {
					self.plugin_id = suite.register_with_aegp(self.descriptor().display_name)?;
				}
				#[cfg(debug_assertions)]
				match self.license.initialize() {
					Ok(()) => log::info!(
						"license: initialize ok; state=[{}]",
						self.license.debug_label().unwrap_or_default()
					),
					Err(e) => log::warn!(
						"license: initialize failed: {e}; state=[{}]",
						self.license.debug_label().unwrap_or_default()
					),
				}
				#[cfg(not(debug_assertions))]
				let _ = self.license.initialize();
			}
			Command::About => {
				let msg = format!(
					"{}\r\nVersion: {}",
					self.descriptor().about_text,
					self.descriptor().version
				);
				out_data.set_return_msg(msg.as_str());
			}
			Command::UpdateParamsUi => {
				self.ensure_ui_rules();
				self.apply_visibility(params, &in_data, &mut out_data)?;
			}
			Command::UserChangedParam { param_index } => {
				self.ensure_ui_rules();
				let changed = params.type_at(param_index);
				for &(param, callback) in E::Params::buttons() {
					if param == changed {
						callback();
					}
				}
				self.apply_visibility(params, &in_data, &mut out_data)?;
			}
			Command::FrameSetup { in_layer, .. } => {
				if !self.license_valid() {
					return Ok(());
				}
				let layer_w = in_layer.width() as u32;
				let layer_h = in_layer.height() as u32;
				let host = host_from_in_data(&in_data);
				let backend = backend_from_cfg();
				let time_seconds = canonical_time_seconds(&in_data);

				let initial_geom = SnapshotGeom {
					layer_w,
					layer_h,
					output_w: layer_w,
					output_h: layer_h,
					ext_x: 0,
					ext_y: 0,
				};
				let state = Self::snapshot_and_ctx(
					params,
					&initial_geom,
					host,
					backend,
					time_seconds,
				)?;
				let ctx = state.ctx(host, backend, 0, time_seconds, false);
				let ext = E::expansion(&ctx);

				let out_w = (layer_w as i32 + ext.left + ext.right).max(1) as u32;
				let out_h = (layer_h as i32 + ext.top + ext.bottom).max(1) as u32;
				if !ext.is_zero() {
					out_data.set_width(out_w);
					out_data.set_height(out_h);
					out_data.set_origin(ae::Point {
						h: ext.left,
						v: ext.top,
					});
				}

				let final_geom = SnapshotGeom {
					layer_w,
					layer_h,
					output_w: out_w,
					output_h: out_h,
					ext_x: ((out_w as i32 - layer_w as i32) / 2).max(0),
					ext_y: ((out_h as i32 - layer_h as i32) / 2).max(0),
				};
				let snapshot = E::Params::snapshot_cpu(params, &final_geom)?;
				let frame_state = FrameState {
					snapshot,
					geom: final_geom,
					time_seconds,
				};
				out_data.set_frame_data::<FrameState<E::Params>>(frame_state);
			}
			Command::FrameSetdown => {
				in_data.destroy_frame_data::<FrameState<E::Params>>();
			}
			Command::Render {
				in_layer,
				mut out_layer,
			} => {
				if !self.license_valid() {
					return Ok(());
				}

		log::debug!(
			"Quality: {:?} | Bit depth: {} | Resolution: {}x{}x{}(stride) | World Type: {:?} | Pixel Format: {:?}",
			in_data.quality(),
			out_layer.bit_depth(),
			out_layer.width(),
			out_layer.height(),
			out_layer.buffer_stride(),
			out_layer.world_type(),
			out_layer.pixel_format()
		);

		if log::log_enabled!(log::Level::Debug) {
			let dbg_bpp =
				crate::cpu::render::compute_bpp(&in_data, &out_layer).unwrap_or(0);
			let dbg_layout =
				crate::cpu::render::pixel_layout_from_format(&in_data, &in_layer);
			let pr_fmt = in_layer.pr_pixel_format();
			let src_pitch_px = if dbg_bpp > 0 {
				in_layer.buffer_stride() as i32 / dbg_bpp as i32
			} else {
				0
			};
			let dst_pitch_px = if dbg_bpp > 0 {
				out_layer.buffer_stride() as i32 / dbg_bpp as i32
			} else {
				0
			};
			let t_sec = canonical_time_seconds(&in_data);
			let local_t_sec = if in_data.time_scale() != 0 {
				in_data.current_time() as f32 / in_data.time_scale() as f32
			} else {
				0.0
			};
			let head = {
				let buf = in_layer.buffer();
				let n = (dbg_bpp as usize).min(buf.len());
				buf[..n]
					.iter()
					.map(|b| format!("{:02x}", *b))
					.collect::<Vec<_>>()
					.join(" ")
			};
			log::debug!(
				"[CPU] computed bpp={dbg_bpp} layout={dbg_layout}(0=RGBA,1=BGRA) flip_y={fy} pr_pixel_format={pr_fmt:?} src_pitch_px={src_pitch_px} dst_pitch_px={dst_pitch_px} t_sec={t_sec:.4} local_t_sec={local_t_sec:.4} current_time={ct} time_step={ts} time_scale={tsc} first_px_bytes=[{head}]",
				fy = in_data.is_premiere() as u32,
				ct = in_data.current_time(),
				ts = in_data.time_step(),
				tsc = in_data.time_scale(),
			);
		}

				let frame_state = in_data
					.frame_data::<FrameState<E::Params>>()
					.ok_or(ae::Error::Generic)?;
				let base = Self::build_invocation_cpu(&in_data, &in_layer, &mut out_layer)?;

				let host = host_from_in_data(&in_data);
				let backend = Backend::Cpu;
				let caps = crate::effect::HostCapabilities::new(host, backend);
				let timing = Timing {
					frame_index: {
						let step = in_data.time_step().max(1);
						(in_data.current_time() / step).max(0) as u32
					},
					time_seconds: frame_state.time_seconds,
					progress: 0.0,
				};
				let geom = Geometry {
					layer_w: frame_state.geom.layer_w,
					layer_h: frame_state.geom.layer_h,
					output_w: frame_state.geom.output_w,
					output_h: frame_state.geom.output_h,
					ext_x: frame_state.geom.ext_x,
					ext_y: frame_state.geom.ext_y,
				};
				let ctx = Ctx::new(&frame_state.snapshot, geom, timing, caps, false);
				let _ = (frame_state, &base);
				self.run_graph(&ctx, &base)?;
			}
			Command::SmartPreRender { mut extra } => {
				if !self.license_valid() {
					return Ok(());
				}
				let req = extra.output_request();
				let req_rect = ae::Rect::from(req.rect);
				let layer_w = req_rect.width().max(1) as u32;
				let layer_h = req_rect.height().max(1) as u32;
				let host = host_from_in_data(&in_data);
				let backend = backend_from_cfg();
				let time_seconds = canonical_time_seconds(&in_data);

				let initial_geom = SnapshotGeom {
					layer_w,
					layer_h,
					output_w: layer_w,
					output_h: layer_h,
					ext_x: 0,
					ext_y: 0,
				};
				let state = Self::snapshot_and_ctx(
					params,
					&initial_geom,
					host,
					backend,
					time_seconds,
				)?;
				let ctx = state.ctx(host, backend, 0, time_seconds, false);
				let ext = E::expansion(&ctx);

				let mut src_request = req;
				src_request.rect = ae::Rect {
					left: req_rect.left - ext.left,
					top: req_rect.top - ext.top,
					right: req_rect.right + ext.right,
					bottom: req_rect.bottom + ext.bottom,
				}
				.into();

				if let Ok(in_result) = extra.callbacks().checkout_layer(
					0,
					0,
					&src_request,
					in_data.current_time(),
					in_data.time_step(),
					in_data.time_scale(),
				) {
					let layer_max = ae::Rect::from(in_result.max_result_rect);
					let layer_result = ae::Rect::from(in_result.result_rect);
					let max_rect = ext.inflate_rect(layer_max);
					let result_rect =
						if ext.is_zero() { layer_result } else { ext.inflate_rect(layer_result) };

					let _ = extra.union_result_rect(result_rect);
					let _ = extra.union_max_result_rect(max_rect);
					if !ext.is_zero() {
						extra.set_returns_extra_pixels(true);
					}
					extra.set_gpu_render_possible(true);

					// Request each declared `#[layer]` param so AE renders it
					// upstream; the pixels are claimed at SmartRender via the
					// matching checkout id. Errors (unassigned layer) are
					// ignored — the pipeline falls back to the main source.
					for (k, layer_id) in E::Params::LAYER_PARAMS.iter().enumerate().take(crate::effect::invocation::MAX_AUX_LAYERS) {
						if let Some(param_idx) = params.index(*layer_id) {
							let _ = extra.callbacks().checkout_layer(
								param_idx as i32,
								Self::aux_checkout_id(k),
								&src_request,
								in_data.current_time(),
								in_data.time_step(),
								in_data.time_scale(),
							);
						}
					}

					let out_w = result_rect.width().max(1) as u32;
					let out_h = result_rect.height().max(1) as u32;
					let src_w = layer_result.width().max(1) as u32;
					let src_h = layer_result.height().max(1) as u32;

					let final_geom = SnapshotGeom {
						layer_w: src_w,
						layer_h: src_h,
						output_w: out_w,
						output_h: out_h,
						ext_x: ((out_w as i32 - src_w as i32) / 2).max(0),
						ext_y: ((out_h as i32 - src_h as i32) / 2).max(0),
					};
					let snapshot = E::Params::snapshot_cpu(params, &final_geom)?;
					let frame_state = FrameState {
						snapshot,
						geom: final_geom,
						time_seconds,
					};
					extra.set_pre_render_data::<FrameState<E::Params>>(frame_state);
				}
			}
			Command::SmartRender { extra } => {
				if !self.license_valid() {
					return Ok(());
				}
				let cb = extra.callbacks();
				let Some(input_world) = cb.checkout_layer_pixels(0)? else {
					return Ok(());
				};
				let render_result = (|| -> Result<(), ae::Error> {
					if let Some(mut output_world) = cb.checkout_output()? {
						let frame_state = extra
							.pre_render_data::<FrameState<E::Params>>()
							.ok_or(ae::Error::Generic)?;
						let mut input_world = input_world;
						let mut base = Self::build_invocation_cpu(
							&in_data,
							&input_world,
							&mut output_world,
						)?;

						// Secondary layer params: claim the pixels requested in
						// SmartPreRender. `_aux_worlds` must outlive run_graph
						// (the bindings hold raw pointers into their buffers).
						let (_aux_worlds, aux_bindings) = Self::checkout_aux_layers(
							&in_data,
							&cb,
							base.bytes_per_pixel,
							base.pixel_layout,
							None,
						);
						base.layers = aux_bindings;

						let host = host_from_in_data(&in_data);
						let backend = Backend::Cpu;
						let caps =
							crate::effect::HostCapabilities::new(host, backend);
						let timing = Timing {
							frame_index: {
								let step = in_data.time_step().max(1);
								(in_data.current_time() / step).max(0) as u32
							},
							time_seconds: frame_state.time_seconds,
							progress: 0.0,
						};
						let geom = Geometry {
							layer_w: frame_state.geom.layer_w,
							layer_h: frame_state.geom.layer_h,
							output_w: frame_state.geom.output_w,
							output_h: frame_state.geom.output_h,
							ext_x: frame_state.geom.ext_x,
							ext_y: frame_state.geom.ext_y,
						};
						let mut ctx = Ctx::new(
							&frame_state.snapshot,
							geom,
							timing,
							caps,
							false,
						);
						ctx.set_layers_present(base.layer_presence());
						let _ = &mut input_world;
						self.run_graph(&ctx, &base)?;
						for (k, _) in &_aux_worlds {
							let _ = cb.checkin_layer_pixels(Self::aux_checkout_id(*k) as u32);
						}
					}
					Ok(())
				})();
				cb.checkin_layer_pixels(0)?;
				render_result?;
			}
			Command::SmartRenderGpu { extra } => {
				if !self.license_valid() {
					return Ok(());
				}
				let cb = extra.callbacks();
				let Some(mut input_world) = cb.checkout_layer_pixels(0)? else {
					return Ok(());
				};
				let render_result = (|| -> Result<(), ae::Error> {
					if let Some(mut output_world) = cb.checkout_output()? {
						let frame_state = extra
							.pre_render_data::<FrameState<E::Params>>()
							.ok_or(ae::Error::Generic)?;
						let mut base = Self::build_invocation_gpu(
							&in_data,
							&mut input_world,
							&mut output_world,
							&extra,
						)?;

						// Secondary layer params (GPU worlds → device pointers).
						// `_aux_worlds` must outlive run_graph.
						let gpu_suite = ae::pf::suites::GPUDevice::new()?;
						let (_aux_worlds, aux_bindings) = Self::checkout_aux_layers(
							&in_data,
							&cb,
							base.bytes_per_pixel,
							base.pixel_layout,
							Some(&gpu_suite),
						);
						base.layers = aux_bindings;

						let host = host_from_in_data(&in_data);
						let backend = base.backend;
						let caps =
							crate::effect::HostCapabilities::new(host, backend);
						let timing = Timing {
							frame_index: {
								let step = in_data.time_step().max(1);
								(in_data.current_time() / step).max(0) as u32
							},
							time_seconds: frame_state.time_seconds,
							progress: 0.0,
						};
						let geom = Geometry {
							layer_w: frame_state.geom.layer_w,
							layer_h: frame_state.geom.layer_h,
							output_w: frame_state.geom.output_w,
							output_h: frame_state.geom.output_h,
							ext_x: frame_state.geom.ext_x,
							ext_y: frame_state.geom.ext_y,
						};
						let mut ctx = Ctx::new(
							&frame_state.snapshot,
							geom,
							timing,
							caps,
							false,
						);
						ctx.set_layers_present(base.layer_presence());
						self.run_graph(&ctx, &base)?;
						for (k, _) in &_aux_worlds {
							let _ = cb.checkin_layer_pixels(Self::aux_checkout_id(*k) as u32);
						}
					}
					Ok(())
				})();
				cb.checkin_layer_pixels(0)?;
				render_result?;
			}
			Command::GpuDeviceSetup { extra } => {
				let what = extra.what_gpu();
				let supported = matches!(what, GpuFramework::Metal | GpuFramework::Cuda);
				if supported {
					out_data.set_out_flag2(ae::OutFlags2::SupportsGpuRenderF32, true);
				}
			}
			Command::GpuDeviceSetdown { .. } => {}
			_ => {}
		}
		Ok(())
	}
}
