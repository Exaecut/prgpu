use after_effects::{self as ae, Error, InData, OutData, ParamFlag, Parameters, Precision};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	Style,
	Amplitude,
	Frequency,
	Phase,
	MotionBlurGroupStart,
	MotionBlur,
	MotionBlurLength,
	MotionBlurSamples,
	MotionBlurGroupEnd,
	Seed,
	RepeatModeX,
	RepeatModeY,
	HorizontalShakeStart,
	HorizontalShakeAmplitude,
	HorizontalShakeFrequency,
	HorizontalShakeEnd,
	VerticalShakeStart,
	VerticalShakeAmplitude,
	VerticalShakeFrequency,
	VerticalShakeEnd,
	TiltGroupStart,
	TiltAmplitude,
	TiltFrequency,
	TiltPhase,
	TiltGroupEnd,
	Debug,
	NoLicense,
}

pub fn setup(params: &mut Parameters<Params>, _in_data: InData, _out_data: OutData) -> Result<(), Error> {
	params.add_customized(
		Params::Help,
		"Documentation",
		ae::ButtonDef::setup(|f| {
			f.set_label("Documentation");
		}),
		|p| {
			p.set_flag(ParamFlag::SUPERVISE, true);
			p.set_flag(ParamFlag::START_COLLAPSED, true);
			-1
		},
	)?;

	if cfg!(debug_assertions) {
		params.add(
			Params::Debug,
			"Debug",
			ae::CheckBoxDef::setup(|f| {
				f.set_label("Debug");
				f.set_default(false);
				f.set_value(false);
			}),
		)?;
	}

	params.add(
		Params::Style,
		"Style",
		ae::PopupDef::setup(|f| {
			f.set_options(&["Perlin Shake", "Wave Shake"]);
			f.set_default(0);
		}),
	)?;

	params.add(
		Params::Amplitude,
		"Amplitude",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(1.0);
			f.set_slider_min(0.0);
			f.set_slider_max(100.0);
			f.set_valid_min(0.0);
			f.set_valid_max(9999.0);
			f.set_precision(ae::Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Frequency,
		"Frequency",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(5.0);
			f.set_slider_min(0.0);
			f.set_slider_max(100.0);
			f.set_valid_min(0.0);
			f.set_valid_max(9999.0);
			f.set_precision(ae::Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Phase,
		"Phase",
		ae::AngleDef::setup(|f| {
			f.set_default(0.0);
			f.set_value(0.0);
		}),
	)?;

	params.add_group(Params::MotionBlurGroupStart, Params::MotionBlurGroupEnd, "Motion Blur", false, |params| {
		params.add(
			Params::MotionBlur,
			"Enable",
			ae::CheckBoxDef::setup(|f| {
				f.set_label("Enable");
				f.set_default(false);
				f.set_value(false);
			}),
		)?;

		params.add(
			Params::MotionBlurLength,
			"Motion Blur Length",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.5);
				f.set_slider_min(1.0);
				f.set_slider_max(10.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_precision(Precision::Hundredths);
			}),
		)?;

		params.add(
			Params::MotionBlurSamples,
			"Motion Blur Samples",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(128.0);
				f.set_slider_min(1.0);
				f.set_slider_max(256.0);
				f.set_valid_min(0.0);
				f.set_valid_max(1024.0);
				f.set_precision(Precision::Integer);
			}),
		)?;

		Ok(())
	})?;

	params.add(
		Params::Seed,
		"Seed",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.0);
			f.set_slider_min(0.0);
			f.set_slider_max(1024.0 * 32.0);
			f.set_valid_min(0.0);
			f.set_valid_max(1024.0 * 64.0);
		}),
	)?;

	params.add(
		Params::RepeatModeX,
		"Repeat Mode X",
		ae::PopupDef::setup(|f| {
			f.set_options(&["None", "Tile", "Mirror"]);
			f.set_default(0);
			f.set_value(0);
		}),
	)?;

	params.add(
		Params::RepeatModeY,
		"Repeat Mode Y",
		ae::PopupDef::setup(|f| {
			f.set_options(&["None", "Tile", "Mirror"]);
			f.set_default(0);
			f.set_value(0);
		}),
	)?;

	params.add_group(Params::HorizontalShakeStart, Params::HorizontalShakeEnd, "Horizontal Shake", true, |params| {
		params.add(
			Params::HorizontalShakeAmplitude,
			"Horizontal Amplitude",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(200.0);
				f.set_slider_min(0.0);
				f.set_slider_max(500.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_precision(2);
			}),
		)?;

		params.add(
			Params::HorizontalShakeFrequency,
			"Horizontal Frequency",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(500.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_precision(2);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(Params::VerticalShakeStart, Params::VerticalShakeEnd, "Vertical Shake", true, |params| {
		params.add(
			Params::VerticalShakeAmplitude,
			"Vertical Amplitude",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(100.0);
				f.set_slider_min(0.0);
				f.set_slider_max(500.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_precision(2);
			}),
		)?;

		params.add(
			Params::VerticalShakeFrequency,
			"Vertical Frequency",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_precision(2);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(Params::TiltGroupStart, Params::TiltGroupEnd, "Tilt Shake", false, |params| {
		params.add(
			Params::TiltAmplitude,
			"Tilt Amplitude",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_curve_tolerance(0.05);
				f.set_precision(2);
			}),
		)?;

		params.add(
			Params::TiltFrequency,
			"Tilt Frequency",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_valid_min(0.0);
				f.set_valid_max(9999.0);
				f.set_precision(2);
			}),
		)?;

		params.add(
			Params::TiltPhase,
			"Tilt Phase",
			ae::AngleDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
			}),
		)?;

		Ok(())
	})?;

	Ok(())
}
