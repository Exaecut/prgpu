#![allow(clippy::drop_non_drop)]
mod datas;
mod params;
mod wgpu_procs;

use datas::*;
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
			wgpu: WgpuProcessing::new(ProcShaderSource::Wgsl(include_str!("../shaders/shake.wgsl"))),
		}
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

			Ok(())
		}
		// Test raw unpatched file for crashes if it crash
	}

	fn handle_command(&mut self, command: Command, in_data: InData, mut out_data: OutData, params: &mut Parameters<Params>) -> Result<(), Error> {
		match command {
			ae::Command::GlobalSetup => {
				if in_data.is_premiere() {
					suites::Utility::new()?.effect_wants_checked_out_frames_to_match_render_pixel_format(in_data.effect_ref())?;
				}

				let _option_button = in_data.effect().set_options_button_name("Infos");

				themis::license::initialize(InitializationOptions {
					product_id: 1,
					authority_mode: AuthorityServerMode::Production,
				});
			}
			ae::Command::About => {
				let msg = format!("Exaecut - Milkshake\r\nVersion: {}", env!("CARGO_PKG_VERSION"));
				out_data.set_return_msg(msg.as_str());
			}
			ae::Command::UserChangedParam { param_index } => {
				if params.type_at(param_index) == Params::Help {
					let _ = webbrowser::open("https://exaecut.io/milkshake/docs");
				}

				if params.type_at(param_index) == Params::NoLicense {
					let _ = webbrowser::open("https://exaecut.io/no-license?id=1");
				}
			}
			ae::Command::FrameSetup { in_layer, .. } => {
				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let amplitude = params.get(Params::Amplitude)?.as_float_slider()?.value() as f32;
				let h_amplitude = params.get(Params::HorizontalShakeAmplitude)?.as_float_slider()?.value() as f32;
				let v_amplitude = params.get(Params::VerticalShakeAmplitude)?.as_float_slider()?.value() as f32;

				let time = in_data.current_time() as f32 / in_data.time_scale() as f32;
				let xframe_size = amplitude * (h_amplitude + v_amplitude);

				let (xframe_x, xframe_y) = (xframe_size * f32::from(in_data.downsample_x()), xframe_size * f32::from(in_data.downsample_y()));

				out_data.set_width((2.0 * xframe_x + in_layer.width() as f32).round() as _);
				out_data.set_height((2.0 * xframe_y + in_layer.height() as f32).round() as _);

				out_data.set_origin(ae::Point {
					h: xframe_x as _,
					v: xframe_y as _,
				});

				out_data.set_frame_data::<KernelParams>(KernelParams::from_params(
					params,
					None,
					None,
					(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
					time,
					in_data,
				)?);
			}
			ae::Command::FrameSetdown { .. } => {
				in_data.destroy_frame_data::<KernelParams>();
			}
			ae::Command::Render { in_layer, mut out_layer } => {
				let in_size = (in_layer.width(), in_layer.height(), in_layer.buffer_stride());
				let out_size = (out_layer.width(), out_layer.height(), out_layer.buffer_stride());
				if !themis::license::is_valid(false) {
					return Ok(());
				}

				let _time = std::time::Instant::now();

				let params = in_data.frame_data::<KernelParams>().unwrap();
				self.wgpu.run_compute(params, in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());
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

				let amplitude = params.get(Params::Amplitude)?.as_float_slider()?.value() as f32;
				let h_amplitude = params.get(Params::HorizontalShakeAmplitude)?.as_float_slider()?.value() as f32;
				let v_amplitude = params.get(Params::VerticalShakeAmplitude)?.as_float_slider()?.value() as f32;

				let time = in_data.current_time() as f32 / in_data.time_scale() as f32;
				let xframe_size = amplitude * (h_amplitude + v_amplitude) + in_data.width().max(in_data.height()) as f32 * 0.5;

				if let Ok(in_result) = extra
					.callbacks()
					.checkout_layer(0, 0, &req, in_data.current_time(), in_data.time_step(), in_data.time_scale())
				{
					let mut res_rect = ae::Rect::from(in_result.max_result_rect);
					let (xframe_x, xframe_y) = (xframe_size * f32::from(in_data.downsample_x()), xframe_size * f32::from(in_data.downsample_y()));
					res_rect.set_origin(ae::Point {
						h: -xframe_x as _,
						v: -xframe_y as _,
					});

					res_rect.set_width((xframe_x + res_rect.width() as f32).round() as _);
					res_rect.set_height((xframe_y + res_rect.height() as f32).round() as _);

					let _ = extra.union_result_rect(res_rect);
					let _ = extra.union_max_result_rect(res_rect);

					extra.set_pre_render_data::<KernelParams>(KernelParams::from_params(
						params,
						Some(xframe_size),
						Some(in_result),
						(f32::from(in_data.downsample_x()), f32::from(in_data.downsample_x())),
						time,
						in_data,
					)?);

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

				let params = extra.pre_render_data::<KernelParams>().unwrap();
				self.wgpu.run_compute(params, in_size, out_size, in_layer.buffer(), out_layer.buffer_mut());

				cb.checkin_layer_pixels(0)?;
			}
			_ => {}
		}

		Ok(())
	}
}

ae::define_effect!(Plugin, (), params::Params);
