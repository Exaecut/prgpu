#![allow(clippy::drop_non_drop)]
mod kernel;
mod params;
mod premiere;

use params::*;
use prgpu::params::SetupParams;
pub use themis::SERVER_PUBLIC_KEY;
use themis::{
	license::InitializationOptions,
	types::{AuthorityServerMode, LicenseState},
};

use after_effects::{self as ae, Error, Parameters};

use crate::kernel::VignetteParams;

pub mod utils {
	pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
		a + (b - a) * t
	}

	pub fn calculate_mipmap_levels(size: (usize, usize, usize)) -> u32 {
		// Take the maximum dimension and compute log2, then add 1 for the base level
		(size.0.max(size.1) as f32).log2().floor() as u32 + 1
	}
}

#[derive(Default)]
struct Plugin {
	plugin_id: ae::aegp::PluginId,
}

#[repr(C)]
struct FrameData {
	main_params: VignetteParams,
}

impl Plugin {
	fn reload_shaders(&mut self) {
		prgpu::gpu::pipeline::hot_reload();
	}
}

impl AdobePluginGlobal for Plugin {
	fn can_load(_host_name: &str, _host_version: &str) -> bool {
		true
	}

	fn params_setup(&self, params: &mut Parameters<Params>, in_data: InData, out_data: OutData) -> Result<(), Error> {
		if themis::license::is_valid(false) {
			let _option_button = in_data.effect().set_options_button_name("Infos");
			Params::setup(params, in_data, out_data)
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

			Params::setup(params, in_data, out_data)
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
					self.plugin_id = suite.register_with_aegp(None, "EXAE Vignette")?;
				}

				themis::license::initialize(InitializationOptions {
					product_id: 43,
					authority_mode: AuthorityServerMode::Production,
					reset: false,
				});
			}
			ae::Command::About => {
				let msg = format!("Exaecut - Vignette\r\nVersion: {}", env!("CARGO_PKG_VERSION"));
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
					let _ = webbrowser::open("https://exaecut.io/feedback/43");
				}

				if params.type_at(param_index) == Params::NoLicense {
					let result = themis::license::initialize(InitializationOptions {
						product_id: 40,
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

				out_data.set_width((in_layer.width() as f32).round() as _);
				out_data.set_height((in_layer.height() as f32).round() as _);

				out_data.set_origin(Point { h: 0 as _, v: 0 as _ });

				let _time = in_data.current_timestamp();

				out_data.set_frame_data::<FrameData>(FrameData {
					main_params: VignetteParams { softness: 0.0, strength: 1.0 },
				});
			}
			ae::Command::FrameSetdown => {
				in_data.destroy_frame_data::<FrameData>();
			}
			ae::Command::Render {
				in_layer,
				#[allow(unused_mut)]
				mut out_layer,
			} => {
				let _in_size = (in_layer.width(), in_layer.height(), in_layer.buffer_stride());
				let _out_size = (out_layer.width(), out_layer.height(), out_layer.buffer_stride());

				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let _out_pixel_format = out_layer.pr_pixel_format();

				let _frame_data = in_data.frame_data::<FrameData>().unwrap();
				// RUN THE SHADER ON CPU. LOAD CPU SHADER MODULE AT RUNTIME RESOLVE THE FUNCTION AND RUN IT.
				// vignette_cpu(config, user_params);
				// 	.run_compute(&frame_data.main_params.as_std140(), in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());
			}
			_ => {}
		}

		Ok(())
	}
}

ae::define_effect!(Plugin, (), params::Params);
