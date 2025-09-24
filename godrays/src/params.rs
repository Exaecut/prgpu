use after_effects::{self as ae, sys::PF_Pixel, Error, InData, OutData, ParamFlag, Parameters, Precision, ValueDisplayFlag};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	ReloadShaders,
	Feedback,

	ClipBounds,
	Threshold,
	ThresholdSmoothness,
	KeyColor,
	KeyColorSensitivity,
	Exposure,
	Decay,
	Bluriness,
	Length,
	LengthMultiplier,
	Center,
	Samples,
	TintColor,
	BlendMode,

	HelperStart,
	PreviewLayer,
	HelperEnd,

	Debug,
	NoLicense,
}

pub fn setup(params: &mut Parameters<Params>, _in_data: InData, _out_data: OutData) -> Result<(), Error> {
	params.add_customized(
		Params::Feedback,
		"Feedback",
		ae::ButtonDef::setup(|f| {
			f.set_label("Feedback");
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

		params.add_customized(
			Params::ReloadShaders,
			"Reload Shaders",
			ae::ButtonDef::setup(|f| {
				f.set_label("Reload Shaders");
			}),
			|p| {
				p.set_flag(ParamFlag::SUPERVISE, true);
				p.set_flag(ParamFlag::START_COLLAPSED, true);
				-1
			},
		)?;
	}

	params.add(
		Params::ClipBounds,
		"Clip bounds",
		ae::CheckBoxDef::setup(|f| {
			f.set_label("Clip bounds");
			f.set_default(true);
			f.set_value(true);
		}),
	)?;

	params.add(
		Params::Threshold,
		"Threshold",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(35.0);
			f.set_valid_min(0.0);
			f.set_valid_max(100.0);
			f.set_slider_min(0.0);
			f.set_slider_max(100.0);
			f.set_display_flags(ValueDisplayFlag::PERCENT);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::ThresholdSmoothness,
		"Threshold smoothness",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.1);
			f.set_valid_min(0.0);
			f.set_valid_max(1.0);
			f.set_slider_min(0.0);
			f.set_slider_max(1.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::KeyColor,
		"Key color",
		ae::ColorDef::setup(|f| {
			f.set_default(PF_Pixel {
				red: 255,
				green: 255,
				blue: 255,
				alpha: 255,
			});

			f.set_value(f.default());
		}),
	)?;

	params.add(
		Params::KeyColorSensitivity,
		"Key color sensitivity",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.3);
			f.set_valid_min(0.0);
			f.set_valid_max(1.0);
			f.set_slider_min(0.0);
			f.set_slider_max(1.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Exposure,
		"Exposure",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.35);
			f.set_slider_min(0.0);
			f.set_slider_max(10.0);
			f.set_valid_min(0.0);
			f.set_valid_max(10.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Decay,
		"Decay",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.95);
			f.set_slider_min(0.0);
			f.set_slider_max(1.0);
			f.set_valid_min(0.0);
			f.set_valid_max(1.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Bluriness,
		"Bluriness",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.0);
			f.set_slider_min(0.0);
			f.set_slider_max(100.0);
			f.set_valid_min(0.0);
			f.set_valid_max(100.0);
			f.set_precision(Precision::Hundredths);
			f.set_display_flags(ValueDisplayFlag::PERCENT);
		}),
	)?;

	params.add(
		Params::Length,
		"Length",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(100.0);
			f.set_slider_min(0.0);
			f.set_slider_max(100.0);
			f.set_valid_min(0.0);
			f.set_valid_max(100.0);
			f.set_exponent(2.0);
			f.set_curve_tolerance(2.0);
		}),
	)?;

	params.add(
		Params::LengthMultiplier,
		"Length multiplier",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(1.0);
			f.set_slider_min(0.0);
			f.set_slider_max(10.0);
			f.set_valid_min(0.0);
			f.set_valid_max(10.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Center,
		"Center",
		ae::PointDef::setup(|f| {
			f.set_default((50.0, 50.0));
			f.set_value(f.default());
			f.set_restrict_bounds(false);
		}),
	)?;

	params.add(
		Params::Samples,
		"Samples (Quality)",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(128.0);
			f.set_value(128.0);
			f.set_slider_min(2.0);
			f.set_slider_max(1024.0);
			f.set_valid_min(2.0);
			f.set_valid_max(1024.0);
			f.set_precision(Precision::Integer);
		}),
	)?;

	params.add(
		Params::TintColor,
		"Tint color",
		ae::ColorDef::setup(|f| {
			f.set_default(PF_Pixel {
				red: 255,
				green: 255,
				blue: 255,
				alpha: 255,
			});
			f.set_value(f.default());
		}),
	)?;

	params.add(
		Params::BlendMode,
		"Blend mode",
		ae::PopupDef::setup(|f| {
			f.set_default(1);
			f.set_value(f.default());
			f.set_options(&["Add", "Screen", "Overlay", "Color Dodge"]);
		}),
	)?;

	params.add_group(Params::HelperStart, Params::HelperEnd, "Helper", true, |params| {
		params.add(
			Params::PreviewLayer,
			"Preview Output",
			ae::PopupDef::setup(|f| {
				f.set_default(1);
				f.set_options(&["Final Output", "Key Color Mask", "Threshold Mask", "Light Shafts only"]);
			}),
		)?;

		Ok(())
	})?;

	Ok(())
}
