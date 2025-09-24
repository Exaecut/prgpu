#![allow(clippy::drop_non_drop)]
mod params;
mod wgpu_procs;

use params::*;
pub use themis::SERVER_PUBLIC_KEY;
use themis::{
	license::InitializationOptions,
	types::{AuthorityServerMode, LicenseState},
};
use wgpu_procs::*;

use after_effects::{self as ae, Error, Parameters};

pub mod utils {
	use crate::ae::log;
	use crate::wgpu_procs::ProcShaderSource;

	pub fn prepare_shader(name: &str) -> ProcShaderSource {
		let shader_path = format!("{}/shaders/{}.wgsl", std::env::current_dir().unwrap().display(), name);
		log::info!("Loading shader on the fly: {}", shader_path);
		let shader = std::fs::read_to_string(shader_path).unwrap();
		ProcShaderSource::Wgsl(shader)
	}

	pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
		a + (b - a) * t
	}

	pub fn calculate_mipmap_levels(size: (usize, usize, usize)) -> u32 {
		// Take the maximum dimension and compute log2, then add 1 for the base level
		(size.0.max(size.1) as f32).log2().floor() as u32 + 1
	}

	pub fn compute_bloom_extent(width: u32, height: u32, mip_count: u32) -> f32 {
		// Compute blur radius: r = 3 * 2 ^ (mip_count-1), or 0 if mip_count = 0
		let blur_radius = if mip_count == 0 {
			0
		} else {
			3 * (1 << (mip_count - 1)) // 2 ^ (mip_count-1) using bitwise shift
		};

		2.0 * (blur_radius as f32) + (width.max(height) as f32 * 0.1)
	}
}

struct Plugin {
	wgpu: wgpu_procs::WgpuProcessing<Std140KernelParams>,
	plugin_id: ae::aegp::PluginId,
}

impl Default for Plugin {
	fn default() -> Self {
		Self {
			wgpu: if cfg!(shader_hotreload) && cfg!(debug_assertions) {
				WgpuProcessing::new(
					utils::prepare_shader("downsample"),
					utils::prepare_shader("upsample"),
					utils::prepare_shader("combine"),
					utils::prepare_shader("blur"),
					utils::prepare_shader("copy_to_16f"),
				)
			} else {
				WgpuProcessing::new(
					ProcShaderSource::Wgsl(include_str!("../shaders/downsample.wgsl").to_string()),
					ProcShaderSource::Wgsl(include_str!("../shaders/upsample.wgsl").to_string()),
					ProcShaderSource::Wgsl(include_str!("../shaders/combine.wgsl").to_string()),
					ProcShaderSource::Wgsl(include_str!("../shaders/blur.wgsl").to_string()),
					ProcShaderSource::Wgsl(include_str!("../shaders/copy_to_16f.wgsl").to_string()),
				)
			},
			plugin_id: ae::aegp::PluginId::default(),
		}
	}
}

#[repr(C)]
struct FrameData {
	downsample_params: DownsampleParams,
	upsample_params: UpsampleParams,
	combine_params: KernelParams,
	blur_params: BlurParams,
}

impl Plugin {
	fn reload_shaders(&mut self) {
		self.wgpu = if cfg!(shader_hotreload) && cfg!(debug_assertions) {
			WgpuProcessing::new(
				utils::prepare_shader("downsample"),
				utils::prepare_shader("upsample"),
				utils::prepare_shader("combine"),
				utils::prepare_shader("blur"),
				utils::prepare_shader("copy_to_16f"),
			)
		} else {
			WgpuProcessing::new(
				ProcShaderSource::Wgsl(include_str!("../shaders/downsample.wgsl").to_string()),
				ProcShaderSource::Wgsl(include_str!("../shaders/upsample.wgsl").to_string()),
				ProcShaderSource::Wgsl(include_str!("../shaders/combine.wgsl").to_string()),
				ProcShaderSource::Wgsl(include_str!("../shaders/blur.wgsl").to_string()),
				ProcShaderSource::Wgsl(include_str!("../shaders/copy_to_16f.wgsl").to_string()),
			)
		};
	}
}

impl AdobePluginGlobal for Plugin {
	fn can_load(_host_name: &str, _host_version: &str) -> bool {
		true
	}

	fn params_setup(&self, params: &mut Parameters<Params>, in_data: InData, out_data: OutData) -> Result<(), Error> {
		if themis::license::is_valid(false) {
			let _option_button = in_data.effect().set_options_button_name("Infos");

			params::setup(params, in_data, out_data)
		} else {
			params.add_customized(
				Params::NoLicense,
				"License check failed",
				ae::ButtonDef::setup(|f| {
					f.set_label(format!("Retry [{}]", LicenseState::debug_string_from_bits(themis::license::get_license_state().bits())).as_str());
				}),
				|p| {
					p.set_flag(ParamFlag::SUPERVISE, true);
					p.set_flag(ParamFlag::START_COLLAPSED, true);
					-1
				},
			)?;

			params::setup(params, in_data, out_data)
		}
	}

	fn handle_command(&mut self, command: Command, in_data: InData, mut out_data: OutData, params: &mut Parameters<Params>) -> Result<(), Error> {
		match command {
			ae::Command::GlobalSetup => {
				log::set_max_level(log::LevelFilter::Info);
				if in_data.is_premiere() {
					suites::Utility::new()?.effect_wants_checked_out_frames_to_match_render_pixel_format(in_data.effect_ref())?;
				}

				let _option_button = in_data.effect().set_options_button_name("Infos");
				if let Ok(suite) = aegp::suites::Utility::new() {
					self.plugin_id = suite.register_with_aegp(None, "EXAE Mindglow")?;
				}

				themis::license::initialize(InitializationOptions {
					product_id: 36,
					authority_mode: AuthorityServerMode::Production,
					reset: false,
				});
			}
			ae::Command::About => {
				let msg = format!("Exaecut - Mindglow\r\nVersion: {}", env!("CARGO_PKG_VERSION"));
				out_data.set_return_msg(msg.as_str());
			}
			ae::Command::UpdateParamsUi => {
				let layer_count = params.get(Params::LayerCount)?.as_slider()?.value() as u32;
				let min_layer_count = params.get(Params::LayerCount)?.as_slider()?.valid_min() as u32;
				let max_layer_count = params.get(Params::LayerCount)?.as_slider()?.valid_max() as u32;
				let mut params_copy = params.clone();

				for i in min_layer_count..max_layer_count {
					let mut layer_param = params_copy.get_mut(Params::LayerSizeIndex(i as usize))?;
					layer_param.set_ui_flag(ParamUIFlags::INVISIBLE, i >= layer_count);
					layer_param.update_param_ui()?;
				}

				if in_data.is_after_effects() {
					let effect = in_data.effect();
					{
						let plugin_id = self.plugin_id;
						let aegp_plugin = effect.aegp_effect(plugin_id)?;

						for i in min_layer_count..max_layer_count {
							let layer_param_stream = aegp_plugin.new_stream_by_index(plugin_id, params.index(Params::LayerSizeIndex(i as usize)).unwrap() as _)?;
							layer_param_stream.set_dynamic_stream_flag(aegp::DynamicStreamFlags::Hidden, false, i >= layer_count)?;
						}
					}
				}

				out_data.set_out_flag(OutFlags::RefreshUi, true);
			}
			ae::Command::UserChangedParam { param_index } => {
				if params.type_at(param_index) == Params::ReloadShaders {
					log::info!("Reloading shaders...");
					self.reload_shaders();
				}

				if params.type_at(param_index) == Params::Help {
					let _ = webbrowser::open("https://exaecut.io/milkshake/docs");
				}

				if params.type_at(param_index) == Params::Feedback {
					let _ = webbrowser::open("https://exaecut.io/feedback/36");
				}

				if params.type_at(param_index) == Params::NoLicense {
					let result = themis::license::initialize(InitializationOptions {
						product_id: 36,
						authority_mode: AuthorityServerMode::Production,
						reset: true,
					});

					if result {
						self.reload_shaders();
						let mut retry_button_param = params.get_mut(Params::NoLicense)?;
						retry_button_param.set_ui_flag(ParamUIFlags::INVISIBLE, true);
						retry_button_param.update_param_ui()?;

						out_data.set_force_rerender();
					}
				}
			}
			ae::Command::FrameSetup { in_layer, .. } => {
				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let mip_count = (params.get(Params::LayerCount)?.as_slider()?.value() as u32).min(utils::calculate_mipmap_levels((
					in_layer.width(),
					in_layer.height(),
					in_layer.buffer_stride(),
				)));

				let xframe_size = if params.get(Params::AllowOOBGlow)?.as_checkbox()?.value() {
					params.get(Params::Radius)?.as_float_slider()?.value() as f32
						+ utils::compute_bloom_extent(in_layer.width() as u32, in_layer.height() as u32, mip_count).floor() as f32
				} else {
					0.0
				};

				let (xframe_x, xframe_y) = (xframe_size * f32::from(in_data.downsample_x()), xframe_size * f32::from(in_data.downsample_y()));

				out_data.set_width((2.0 * xframe_x + in_layer.width() as f32).round() as _);
				out_data.set_height((2.0 * xframe_y + in_layer.height() as f32).round() as _);

				out_data.set_origin(Point {
					h: xframe_x as _,
					v: xframe_y as _,
				});

				let time = in_data.current_timestamp();

				out_data.set_frame_data::<FrameData>(FrameData {
					downsample_params: DownsampleParams::from_params(params, time as f32, in_data)?,
					upsample_params: UpsampleParams::from_params(params, time as f32, in_data)?,
					combine_params: KernelParams::from_params(
						params,
						time as f32,
						(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
						in_data,
						xframe_size,
					)?,
					blur_params: BlurParams::from_params(params)?,
				});
			}
			ae::Command::FrameSetdown => {
				in_data.destroy_frame_data::<FrameData>();
			}
			ae::Command::Render { in_layer, mut out_layer } => {
				let in_size = (in_layer.width(), in_layer.height(), in_layer.buffer_stride());
				let out_size = (out_layer.width(), out_layer.height(), out_layer.buffer_stride());
				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let _out_pixel_format = out_layer.pr_pixel_format();

				let mip_length = params.get(Params::LayerCount)?.as_slider()?.value();

				let frame_data = in_data.frame_data::<FrameData>().unwrap();
				self.wgpu.run_compute(
					params,
					&frame_data.combine_params,
					&frame_data.downsample_params,
					&frame_data.upsample_params,
					&frame_data.blur_params,
					mip_length as usize,
					in_size,
					out_size,
					in_layer.buffer(),
					out_layer.buffer_mut(),
				);
			}
			ae::Command::SmartPreRender { mut extra } => {
				let req = extra.output_request();

				if !themis::license::is_valid(false) {
					if let Ok(in_result) = extra
						.callbacks()
						.checkout_layer(0, 0, &req, in_data.current_time(), in_data.time_step(), in_data.time_scale())
					{
						let _ = extra.union_max_result_rect(ae::Rect::from(in_result.max_result_rect));
					}

					return Ok(());
				}

				if let Ok(in_result) = extra
					.callbacks()
					.checkout_layer(0, 0, &req, in_data.current_time(), in_data.time_step(), in_data.time_scale())
				{
					let mut res_rect = ae::Rect::from(in_result.max_result_rect);

					let mip_count = (params.get(Params::LayerCount)?.as_slider()?.value() as u32).min(utils::calculate_mipmap_levels((
						res_rect.width() as usize,
						res_rect.height() as usize,
						1,
					)));

					let xframe_size = if params.get(Params::AllowOOBGlow)?.as_checkbox()?.value() {
						params.get(Params::Radius)?.as_float_slider()?.value() as f32
							+ utils::compute_bloom_extent(res_rect.width() as u32, res_rect.height() as u32, mip_count).floor() as f32
					} else {
						0.0
					};

					let (xframe_x, xframe_y) = (xframe_size * f32::from(in_data.downsample_x()), xframe_size * f32::from(in_data.downsample_y()));

					res_rect.set_origin(ae::Point {
						h: -xframe_x.floor() as i32 + res_rect.origin().h,
						v: -xframe_y.floor() as i32 + res_rect.origin().v,
					});

					res_rect.set_width((xframe_x + res_rect.width() as f32).round() as _);
					res_rect.set_height((xframe_y + res_rect.height() as f32).round() as _);

					let _ = extra.union_result_rect(res_rect);
					let _ = extra.union_max_result_rect(res_rect);

					let time = in_data.current_timestamp();

					extra.set_pre_render_data::<FrameData>(FrameData {
						downsample_params: DownsampleParams::from_params(params, time as f32, in_data)?,
						upsample_params: UpsampleParams::from_params(params, time as f32, in_data)?,
						combine_params: KernelParams::from_params(
							params,
							time as f32,
							(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
							in_data,
							xframe_size,
						)?,
						blur_params: BlurParams::from_params(params)?,
					});

					extra.set_returns_extra_pixels(true);
				}
			}
			ae::Command::SmartRender { extra } => {
				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let cb = extra.callbacks();
				let Some(in_layer) = cb.checkout_layer_pixels(0)? else {
					return Ok(());
				};

				let Some(mut out_layer) = cb.checkout_output()? else {
					return Ok(());
				};

				let in_size = (in_layer.width(), in_layer.height(), in_layer.buffer_stride());
				let out_size = (out_layer.width(), out_layer.height(), out_layer.buffer_stride());

				log::info!("Smart render: {:?}x{:?} -> {:?}x{:?}", in_size.0, in_size.1, out_size.0, out_size.1);

				let mip_length = params.get(Params::LayerCount)?.as_slider()?.value() as f32;

				let _time = std::time::Instant::now();

				let frame_data = extra.pre_render_data::<FrameData>().unwrap();
				self.wgpu.run_compute(
					params,
					&frame_data.combine_params,
					&frame_data.downsample_params,
					&frame_data.upsample_params,
					&frame_data.blur_params,
					mip_length.floor() as usize,
					in_size,
					out_size,
					in_layer.buffer(),
					out_layer.buffer_mut(),
				);

				cb.checkin_layer_pixels(0)?;
			}
			_ => {}
		}

		Ok(())
	}
}

ae::define_effect!(Plugin, (), params::Params);
