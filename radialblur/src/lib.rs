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

struct Plugin {
	wgpu: wgpu_procs::WgpuProcessing<KernelParams>,
}

impl Default for Plugin {
	fn default() -> Self {
		Self {
			wgpu: if cfg!(debug_assertions) {
				let main_shader_path = format!("{}/shaders/main.wgsl", std::env::current_dir().unwrap().display());
				log::info!("Loading shaders on the fly. \n{main_shader_path}");

				let main_pass_path = std::fs::read_to_string(main_shader_path.clone()).unwrap();
				let main_pass = ProcShaderSource::Wgsl(&main_pass_path);

				WgpuProcessing::new(main_pass)
			} else {
				WgpuProcessing::new(ProcShaderSource::Wgsl(include_str!("../shaders/main.wgsl")))
			},
		}
	}
}

#[repr(C)]
struct FrameData {
	main_params: KernelParams,
}

impl Plugin {
	fn reload_shaders(&mut self) {
		self.wgpu = if cfg!(debug_assertions) {
			let main_shader_path = format!("{}/shaders/main.wgsl", std::env::current_dir().unwrap().display());
			log::info!("Loading shaders on the fly. \n{main_shader_path}");

			let main_pass_path = std::fs::read_to_string(main_shader_path.clone()).unwrap();
			let main_pass = ProcShaderSource::Wgsl(&main_pass_path);

			WgpuProcessing::new(main_pass)
		} else {
			WgpuProcessing::new(ProcShaderSource::Wgsl(include_str!("../shaders/main.wgsl")))
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
				if in_data.is_premiere() {
					suites::Utility::new()?.effect_wants_checked_out_frames_to_match_render_pixel_format(in_data.effect_ref())?;
				}

				let _option_button = in_data.effect().set_options_button_name("Infos");

				themis::license::initialize(InitializationOptions {
					product_id: 35,
					authority_mode: AuthorityServerMode::Production,
					reset: false,
				});
			}
			ae::Command::About => {
				let msg = format!("Exaecut - Radial blur\r\nVersion: {}", env!("CARGO_PKG_VERSION"));
				out_data.set_return_msg(msg.as_str());
			}
			ae::Command::UserChangedParam { param_index } => {
				if params.type_at(param_index) == Params::ReloadShaders {
					log::info!("Reloading shaders...");
					self.reload_shaders();
				}

				if params.type_at(param_index) == Params::Help {
					let _ = webbrowser::open("https://exaecut.io/docs/35");
				}

				if params.type_at(param_index) == Params::Feedback {
					let _ = webbrowser::open("https://exaecut.io/feedback/35");
				}

				if params.type_at(param_index) == Params::NoLicense {
					let result = themis::license::initialize(InitializationOptions {
						product_id: 35,
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

				out_data.set_width(in_layer.width() as _);
				out_data.set_height(in_layer.height() as _);

				out_data.set_origin(ae::Point { h: 0, v: 0 });

				let time = in_data.current_timestamp();

				out_data.set_frame_data::<FrameData>(FrameData {
					main_params: KernelParams::from_params(
						params,
						time as f32,
						(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
						in_data,
						None,
						None,
					)?,
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
				println!("out pixel format: {_out_pixel_format:?}");

				let params = in_data.frame_data::<FrameData>().unwrap();
				self.wgpu
					.run_compute(&params.main_params, in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());
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
					let outer_spread = params.get(Params::OuterSpread)?.as_float_slider()?.value() as f32;

					let xframe = (res_rect.width().max(res_rect.height()) as f32) * outer_spread;

					res_rect.set_origin(ae::Point {
						h: res_rect.origin().h - (xframe.floor()) as i32,
						v: res_rect.origin().v - (xframe.floor()) as i32,
					});

					res_rect.set_width(res_rect.width() + (xframe.floor()) as i32);
					res_rect.set_height(res_rect.height() + (xframe.floor()) as i32);

					let _ = extra.union_result_rect(res_rect);
					let _ = extra.union_max_result_rect(res_rect);

					let time = in_data.current_timestamp();

					extra.set_pre_render_data::<FrameData>(FrameData {
						main_params: KernelParams::from_params(
							params,
							time as f32,
							(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
							in_data,
							Some(res_rect),
							Some(xframe),
						)?,
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

				let _time = std::time::Instant::now();

				let params = extra.pre_render_data::<FrameData>().unwrap();
				self.wgpu
					.run_compute(&params.main_params, in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());

				cb.checkin_layer_pixels(0)?;
			}
			_ => {}
		}

		Ok(())
	}
}

ae::define_effect!(Plugin, (), params::Params);
