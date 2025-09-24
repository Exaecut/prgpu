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
		log::info!("Loading shader on the fly: {shader_path}");
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
}

struct Plugin {
	wgpu: wgpu_procs::WgpuProcessing<Std140KernelParams>,
	plugin_id: ae::aegp::PluginId,
}

impl Default for Plugin {
	fn default() -> Self {
		Self {
			wgpu: if cfg!(shader_hotreload) && cfg!(debug_assertions) {
				WgpuProcessing::new(utils::prepare_shader("main"))
			} else {
				WgpuProcessing::new(ProcShaderSource::Wgsl(include_str!("../shaders/main.wgsl").to_string()))
			},
			plugin_id: ae::aegp::PluginId::default(),
		}
	}
}

#[repr(C)]
struct FrameData {
	main_params: KernelParams,
}

impl Plugin {
	fn reload_shaders(&mut self) {
		self.wgpu = if cfg!(shader_hotreload) && cfg!(debug_assertions) {
			WgpuProcessing::new(utils::prepare_shader("main"))
		} else {
			WgpuProcessing::new(ProcShaderSource::Wgsl(include_str!("../shaders/main.wgsl").to_string()))
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
					f.set_label(format!("Check status [{}]", LicenseState::debug_string_from_bits(themis::license::get_license_state().bits())).as_str());
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
					self.plugin_id = suite.register_with_aegp(None, "EXAE Echo")?;
				}

				themis::license::initialize(InitializationOptions {
					product_id: 46,
					authority_mode: AuthorityServerMode::Production,
					reset: false,
				});
			}
			ae::Command::About => {
				let msg = format!("Exaecut - Echo\r\nVersion: {}", env!("CARGO_PKG_VERSION"));
				out_data.set_return_msg(msg.as_str());
			}
			ae::Command::UpdateParamsUi => {
				let mut _params_copy = params.clone();

				if in_data.is_after_effects() {
					let effect = in_data.effect();
					{
						let plugin_id = self.plugin_id;
						let _aegp_plugin = effect.aegp_effect(plugin_id)?;

						// ...
					}
				}

				out_data.set_out_flag(OutFlags::RefreshUi, true);
			}
			ae::Command::UserChangedParam { param_index } => {
				if params.type_at(param_index) == Params::ReloadShaders {
					log::info!("Reloading shaders...");
					self.reload_shaders();
				}

				if params.type_at(param_index) == Params::Feedback {
					let _ = webbrowser::open("https://exaecut.io/feedback/46");
				}

				if params.type_at(param_index) == Params::NoLicense {
					let _ = webbrowser::open("https://exaecut.io/no-license?id=46");
				}
			}
			ae::Command::FrameSetup { in_layer, .. } => {
				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let extension = 0;

				out_data.set_width((in_layer.width() as i32 + extension) as _);
				out_data.set_height((in_layer.height() as i32 + extension) as _);

				out_data.set_origin(ae::Point { h: 0 as _, v: 0 as _ });

				out_data.set_frame_data::<FrameData>(FrameData {
					main_params: KernelParams::from_params(
						params,
						(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
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

				let frame_data = in_data.frame_data::<FrameData>().unwrap();
				self.wgpu
					.run_compute(&frame_data.main_params, in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());
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

					let extension = 0;

					res_rect.set_origin(ae::Point {
						h: res_rect.origin().h - extension,
						v: res_rect.origin().v - extension,
					});

					res_rect.set_width((res_rect.width() + extension) as _);
					res_rect.set_height((res_rect.height() + extension) as _);

					let _ = extra.union_result_rect(res_rect);
					let _ = extra.union_max_result_rect(res_rect);

					extra.set_pre_render_data::<FrameData>(FrameData {
						main_params: KernelParams::from_params(
							params,
							(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
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

				log::info!("Smart render: {:?}x{:?} -> {:?}x{:?}", in_size.0, in_size.1, out_size.0, out_size.1);

				let _time = std::time::Instant::now();

				let frame_data = extra.pre_render_data::<FrameData>().unwrap();
				self.wgpu
					.run_compute(&frame_data.main_params, in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());

				cb.checkin_layer_pixels(0)?;
			}
			_ => {}
		}

		Ok(())
	}
}

ae::define_effect!(Plugin, (), params::Params);
