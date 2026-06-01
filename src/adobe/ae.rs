//! After Effects PF adapter: drives every selector through the
//! `Effect` trait so per-effect crates stop hand-writing `handle_command`.
//!
//! Selector mapping:
//!
//! | AE selector             | Effect method                                      |
//! |-------------------------|----------------------------------------------------|
//! | `Cmd_GlobalSetup`       | `Effect::descriptor` + `Effect::License::initialize` |
//! | `Cmd_About`             | descriptor `about_text`                            |
//! | `Cmd_ParamsSetup`       | `Effect::params`                                   |
//! | `Cmd_UpdateParamsUi`    | `Effect::ui` + visibility replay                   |
//! | `Cmd_UserChangedParam`  | `Effect::ui` + matching `on_click`                 |
//! | `Cmd_FrameSetup`        | `Effect::expansion` + `Effect::frame_data`         |
//! | `Cmd_FrameSetdown`      | drop frame data                                    |
//! | `Cmd_Render`            | graph execution (CPU)                              |
//! | `Cmd_SmartPreRender`    | expansion + frame data                             |
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

use after_effects::{self as ae, Command, GpuFramework, InData, OutData, Parameters};
use after_effects::aegp;
use parking_lot::Mutex;

use crate::effect::descriptor::install_descriptor_pixel_formats;
use crate::effect::frame_context::{FrameDataContext, HostBackend};
use crate::effect::host::{Host, RenderKind};
use crate::effect::params_api::{ActionContext, ActionRule, ParamApi, VisibilityRule};
use crate::effect::{Effect, EffectDescriptor, FrameBinding, InvocationBase, LicenseGate, PixelLayout};
use crate::graph::{execute::execute as run_graph, RenderGraph};
use crate::types::Backend;

/// Cached visibility/action rules collected by `Effect::ui`.
struct UiState<E: Effect> {
	visibility: Vec<VisibilityRule<E::Params>>,
	actions: Vec<ActionRule<E::Params>>,
}

/// AE PF adapter. Implements [`AdobePluginGlobal`] over the [`Effect`] trait
/// so `ae::define_effect!(Plugin, (), Params)` can register a plugin whose
/// only declarative content lives in `impl Effect for MyEffect`.
pub struct EffectAdapter<E: Effect> {
	license: E::License,
	graph: OnceLock<RenderGraph<E::FrameData>>,
	descriptor: OnceLock<EffectDescriptor>,
	plugin_id: aegp::PluginId,
	ui_cache: Mutex<Option<UiState<E>>>,
}

impl<E: Effect> Default for EffectAdapter<E> {
	fn default() -> Self {
		Self {
			license: E::License::default(),
			graph: OnceLock::new(),
			descriptor: OnceLock::new(),
			plugin_id: aegp::PluginId::default(),
			ui_cache: Mutex::new(None),
		}
	}
}

impl<E: Effect> EffectAdapter<E> {
	fn descriptor(&self) -> &EffectDescriptor {
		self.descriptor.get_or_init(E::descriptor)
	}

	fn graph(&self) -> &RenderGraph<E::FrameData> {
		self.graph.get_or_init(|| {
			let mut g = RenderGraph::new();
			E::pipeline(&mut g);
			g
		})
	}

	fn collect_ui_rules(in_data: &InData, out_data: OutData, params: &mut Parameters<E::Params>) -> Result<UiState<E>, ae::Error> {
		// SAFETY: `ParamApi::new` requires a `&mut Parameters<P>` with the
		// same lifetime as `'a`. We use the AE-supplied params for the
		// duration of `Effect::ui` only; nothing escapes.
		let mut api: ParamApi<E::Params> = unsafe {
			let params_ptr: *mut Parameters<E::Params> = params as *mut _;
			let in_data_clone = in_data.clone();
			ParamApi::new(&mut *params_ptr, in_data_clone, out_data)
		};
		E::ui(&mut api)?;
		let (vis, act) = api.into_rules();
		Ok(UiState { visibility: vis, actions: act })
	}

	fn apply_visibility(&self, params: &mut Parameters<E::Params>, in_data: &InData, out_data: &mut OutData, ui_state: &UiState<E>) -> Result<(), ae::Error> {
		let host = if in_data.is_premiere() { Host::Premiere } else { Host::AfterEffects };
		// Backend at param-tick is unknown; the visibility predicates only
		// see `Capability::FrameExpansion` (tied to host) so CPU is a safe
		// stand-in for the backend position here.
		let caps = crate::effect::HostCapabilities::new(host, Backend::Cpu);

		let mut params_copy = params.clone();
		let mut visible_map: Vec<(E::Params, bool)> = Vec::with_capacity(ui_state.visibility.len());
		for rule in &ui_state.visibility {
			let visible = (rule.predicate)(params, caps);
			visible_map.push((rule.param, visible));
			if let Ok(mut p) = params_copy.get_mut(rule.param) {
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
							let _ = stream.set_dynamic_stream_flag(aegp::DynamicStreamFlags::Hidden, false, !*visible);
						}
					}
				}
			}
		}

		out_data.set_out_flag(ae::OutFlags::RefreshUi, true);
		Ok(())
	}

	fn build_frame_data_cpu(in_data: &InData, params: &Parameters<E::Params>, in_layer: Option<&ae::Layer>, out_w: u32, out_h: u32) -> Result<E::FrameData, ae::Error> {
		let host = if in_data.is_premiere() { Host::Premiere } else { Host::AfterEffects };
		let backend = Backend::Cpu;
		let render_kind = if in_data.is_premiere() { RenderKind::PremiereGpuEffect } else { RenderKind::AeSmartRenderCpu };
		let layer_w = in_layer.map(|l| l.width() as u32).unwrap_or(out_w);
		let layer_h = in_layer.map(|l| l.height() as u32).unwrap_or(out_h);

		let frame_index = {
			let step = in_data.time_step().max(1);
			(in_data.current_time() / step).max(0) as u32
		};
		let time_seconds = if in_data.time_scale() != 0 {
			in_data.current_time() as f32 / in_data.time_scale() as f32
		} else {
			0.0
		};

		let ctx = FrameDataContext {
			host,
			backend,
			render_kind,
			inner: HostBackend::Cpu { params, is_premiere: in_data.is_premiere() },
			layer_width: layer_w,
			layer_height: layer_h,
			output_width: out_w,
			output_height: out_h,
			frame_index,
			time_seconds,
			progress: 0.0,
		};
		E::frame_data(ctx)
	}

	fn build_invocation_cpu(in_data: &InData, in_layer: &ae::Layer, out_layer: &mut ae::Layer) -> Result<InvocationBase, ae::Error> {
		let bpp = crate::cpu::render::compute_bpp(in_data, out_layer)?;
		let pixel_layout = PixelLayout::from_u32(crate::cpu::render::pixel_layout_from_format(in_data, in_layer));

		let in_w = in_layer.width() as u32;
		let in_h = in_layer.height() as u32;
		let out_w = out_layer.width() as u32;
		let out_h = out_layer.height() as u32;

		let in_ptr = in_layer.buffer().as_ptr() as *mut c_void;
		let out_ptr = out_layer.buffer_mut().as_mut_ptr() as *mut c_void;
		let src_pitch = in_layer.buffer_stride() as i32 / bpp as i32;
		let dest_pitch = out_layer.buffer_stride() as i32 / bpp as i32;

		let host = if in_data.is_premiere() { Host::Premiere } else { Host::AfterEffects };
		let render_kind = if in_data.is_premiere() { RenderKind::PremiereGpuEffect } else { RenderKind::AeSmartRenderCpu };

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
			time: if in_data.time_scale() != 0 { in_data.current_time() as f32 / in_data.time_scale() as f32 } else { 0.0 },
			progress: 0.0,
			render_generation: 0,
			main_source: main,
			incoming_source: None,
			outgoing_source: None,
			output,
		})
	}

	fn build_invocation_gpu(in_data: &InData, in_layer: &mut ae::Layer, out_layer: &mut ae::Layer, extra: &ae::pf::SmartRenderExtra) -> Result<InvocationBase, ae::Error> {
		let gpu_suite = ae::pf::suites::GPUDevice::new()?;
		let device_index = extra.device_index();
		let info = gpu_suite.device_info(in_data.effect_ref(), device_index)?;

		let src_mem = gpu_suite.gpu_world_data(in_data.effect_ref(), &mut *in_layer)?;
		let dst_mem = gpu_suite.gpu_world_data(in_data.effect_ref(), &mut *out_layer)?;

		// AE GPU world's pixel format isn't reflected in `world_type()`;
		// detect via `pixel_format()` and fall back to CPU sniffing.
		let bpp = match out_layer.pixel_format() {
			Ok(ae::pf::PixelFormat::GpuBgra128) | Ok(ae::pf::PixelFormat::Argb128) => 16u32,
			Ok(ae::pf::PixelFormat::Argb64) => 8u32,
			_ => crate::cpu::render::compute_bpp(in_data, out_layer)?,
		};
		let pixel_layout = PixelLayout::from_u32(crate::cpu::render::pixel_layout_from_format(in_data, in_layer));

		let in_w = in_layer.width() as u32;
		let in_h = in_layer.height() as u32;
		let out_w = out_layer.width() as u32;
		let out_h = out_layer.height() as u32;
		let src_pitch = in_layer.buffer_stride() as i32 / bpp as i32;
		let dest_pitch = out_layer.buffer_stride() as i32 / bpp as i32;

		// Metal: device handle is the device pointer. CUDA: device handle is
		// the context pointer.
		#[cfg(gpu_backend = "metal")]
		let device_ptr = info.devicePV;
		#[cfg(gpu_backend = "cuda")]
		let device_ptr = info.contextPV;
		#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
		let device_ptr = std::ptr::null_mut();

		let backend = match extra.what_gpu() {
			GpuFramework::Cuda => Backend::Cuda,
			GpuFramework::Metal => Backend::Metal,
			GpuFramework::OpenCl => Backend::OpenCL,
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
			host: if in_data.is_premiere() { Host::Premiere } else { Host::AfterEffects },
			backend,
			render_kind: RenderKind::AeSmartRenderGpu,
			device_handle: device_ptr as *mut c_void,
			context_handle: if info.contextPV.is_null() { None } else { Some(info.contextPV as *mut c_void) },
			command_queue_handle: info.command_queuePV as *mut c_void,
			bytes_per_pixel: bpp,
			pixel_layout,
			time: if in_data.time_scale() != 0 { in_data.current_time() as f32 / in_data.time_scale() as f32 } else { 0.0 },
			progress: 0.0,
			render_generation: frame_index as u64,
			main_source: main,
			incoming_source: None,
			outgoing_source: None,
			output,
		})
	}

	fn run_graph(&self, frame_data: &E::FrameData, base: &InvocationBase) -> Result<(), ae::Error> {
		run_graph(self.graph(), frame_data, base).map_err(|_| ae::Error::Generic)
	}
}

impl<E: Effect> EffectAdapter<E> {
	/// AE PF `params_setup` selector. Call from the user's
	/// `AdobePluginGlobal::params_setup` impl (generated by
	/// `define_effect!`).
	pub fn params_setup(&self, params: &mut Parameters<E::Params>, in_data: InData, out_data: OutData) -> Result<(), ae::Error> {
		if let Some(label) = self.descriptor().options_button {
			let _ = in_data.effect().set_options_button_name(label);
		}
		E::params(params, in_data, out_data)?;

		// Capture the real discriminant→host-param-index map from the registration
		// order so the Premiere GPU path (which indexes params positionally) reads
		// the same param the CPU path resolves through `Parameters::index`. Without
		// this, registering a param out of discriminant order (e.g. the license
		// button first) shifts every GPU param read. See `params::get_param`.
		let gpu_indices: std::collections::HashMap<usize, usize> =
			params.map.iter().map(|(p, info)| ((*p).into(), info.index)).collect();
		crate::params::register_gpu_param_indices::<E::Params>(gpu_indices);
		Ok(())
	}

	/// AE PF `handle_command` selector. Call from the user's
	/// `AdobePluginGlobal::handle_command` impl.
	pub fn handle_command(&mut self, command: Command, in_data: InData, mut out_data: OutData, params: &mut Parameters<E::Params>) -> Result<(), ae::Error> {
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
				let _ = self.license.initialize();
			}
			Command::About => {
				let msg = format!("{}\r\nVersion: {}", self.descriptor().about_text, self.descriptor().version);
				out_data.set_return_msg(msg.as_str());
			}
			Command::UpdateParamsUi => {
				if let Ok(state) = Self::collect_ui_rules(&in_data, out_data.clone(), params) {
					self.apply_visibility(params, &in_data, &mut out_data, &state)?;
					*self.ui_cache.lock() = Some(state);
				}
			}
			Command::UserChangedParam { param_index } => {
				let changed = params.type_at(param_index);
				let mut hot_reload = false;
				let cache_state = self.ui_cache.lock().take();
				let state = match cache_state {
					Some(s) => s,
					None => Self::collect_ui_rules(&in_data, out_data.clone(), params)?,
				};
				for rule in &state.actions {
					if rule.param == changed {
						let mut ctx = ActionContext::new();
						let _ = (rule.callback)(&mut ctx);
						if ctx.hot_reload_shaders {
							hot_reload = true;
						}
					}
				}
				*self.ui_cache.lock() = Some(state);
				if hot_reload {
					crate::gpu::pipeline::hot_reload();
				}
			}
			Command::FrameSetup { in_layer, .. } => {
				if !self.license.is_valid() {
					return Ok(());
				}
				let layer_w = in_layer.width() as u32;
				let layer_h = in_layer.height() as u32;
				let exp_ctx = FrameDataContext {
					host: if in_data.is_premiere() { Host::Premiere } else { Host::AfterEffects },
					backend: Backend::Cpu,
					render_kind: if in_data.is_premiere() { RenderKind::PremiereGpuEffect } else { RenderKind::AeSmartRenderCpu },
					inner: HostBackend::Cpu { params, is_premiere: in_data.is_premiere() },
					layer_width: layer_w,
					layer_height: layer_h,
					output_width: layer_w,
					output_height: layer_h,
					frame_index: 0,
					time_seconds: 0.0,
					progress: 0.0,
				};
				let ext = E::expansion(exp_ctx)?;

				let out_w = (layer_w as i32 + ext.left + ext.right).max(1) as u32;
				let out_h = (layer_h as i32 + ext.top + ext.bottom).max(1) as u32;
				if !ext.is_zero() {
					out_data.set_width(out_w);
					out_data.set_height(out_h);
					out_data.set_origin(ae::Point { h: ext.left, v: ext.top });
				}

				let frame_data = Self::build_frame_data_cpu(&in_data, params, Some(&in_layer), out_w, out_h)?;
				out_data.set_frame_data::<E::FrameData>(frame_data);
			}
			Command::FrameSetdown => {
				in_data.destroy_frame_data::<E::FrameData>();
			}
			Command::Render { mut in_layer, mut out_layer } => {
				if !self.license.is_valid() {
					return Ok(());
				}
				let frame_data = in_data.frame_data::<E::FrameData>().ok_or(ae::Error::Generic)?;
				let base = Self::build_invocation_cpu(&in_data, &in_layer, &mut out_layer)?;
				let _ = (frame_data, &base);
				self.run_graph(frame_data, &base)?;
			}
			Command::SmartPreRender { mut extra } => {
				if !self.license.is_valid() {
					return Ok(());
				}
				let req = extra.output_request();
				let req_rect = ae::Rect::from(req.rect);
				let layer_w = req_rect.width().max(1) as u32;
				let layer_h = req_rect.height().max(1) as u32;
				let exp_ctx = FrameDataContext {
					host: if in_data.is_premiere() { Host::Premiere } else { Host::AfterEffects },
					backend: Backend::Cpu,
					render_kind: RenderKind::AeSmartRenderCpu,
					inner: HostBackend::Cpu { params, is_premiere: in_data.is_premiere() },
					layer_width: layer_w,
					layer_height: layer_h,
					output_width: layer_w,
					output_height: layer_h,
					frame_index: 0,
					time_seconds: 0.0,
					progress: 0.0,
				};
				let ext = E::expansion(exp_ctx)?;

				let mut src_request = req;
				src_request.rect = ae::Rect {
					left: req_rect.left - ext.left,
					top: req_rect.top - ext.top,
					right: req_rect.right + ext.right,
					bottom: req_rect.bottom + ext.bottom,
				}
				.into();

				if let Ok(in_result) = extra.callbacks().checkout_layer(0, 0, &src_request, in_data.current_time(), in_data.time_step(), in_data.time_scale()) {
					let layer_max = ae::Rect::from(in_result.max_result_rect);
					let layer_result = ae::Rect::from(in_result.result_rect);
					let max_rect = ext.inflate_rect(layer_max);
					let result_rect = if ext.is_zero() { layer_result } else { ext.inflate_rect(layer_result) };

					let _ = extra.union_result_rect(result_rect);
					let _ = extra.union_max_result_rect(max_rect);
					if !ext.is_zero() {
						extra.set_returns_extra_pixels(true);
					}
					extra.set_gpu_render_possible(true);

					let out_w = result_rect.width().max(1) as u32;
					let out_h = result_rect.height().max(1) as u32;
					let frame_data = Self::build_frame_data_cpu(&in_data, params, None, out_w, out_h)?;
					extra.set_pre_render_data::<E::FrameData>(frame_data);
				}
			}
			Command::SmartRender { extra } => {
				if !self.license.is_valid() {
					return Ok(());
				}
				let cb = extra.callbacks();
				let Some(input_world) = cb.checkout_layer_pixels(0)? else { return Ok(()) };
				let render_result = (|| -> Result<(), ae::Error> {
					if let Some(mut output_world) = cb.checkout_output()? {
						let frame_data = extra.pre_render_data::<E::FrameData>().ok_or(ae::Error::Generic)?;
						let mut input_world = input_world;
						let base = Self::build_invocation_cpu(&in_data, &input_world, &mut output_world)?;
						let _ = &mut input_world;
						self.run_graph(frame_data, &base)?;
					}
					Ok(())
				})();
				cb.checkin_layer_pixels(0)?;
				render_result?;
			}
			Command::SmartRenderGpu { extra } => {
				if !self.license.is_valid() {
					return Ok(());
				}
				let cb = extra.callbacks();
				let Some(mut input_world) = cb.checkout_layer_pixels(0)? else { return Ok(()) };
				let render_result = (|| -> Result<(), ae::Error> {
					if let Some(mut output_world) = cb.checkout_output()? {
						let frame_data = extra.pre_render_data::<E::FrameData>().ok_or(ae::Error::Generic)?;
						let base = Self::build_invocation_gpu(&in_data, &mut input_world, &mut output_world, &extra)?;
						self.run_graph(frame_data, &base)?;
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
