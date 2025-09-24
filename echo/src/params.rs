use after_effects::{self as ae, sys::PF_Pixel, Error, InData, OutData, ParamFlag, Parameters, Precision, ValueDisplayFlag};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	ReloadShaders,
	Feedback,

	ClipBounds,
	Echo,

	TransformStart,
	TransformPosition,
	TransformRotation,
	TransformScale,
	TransformEnd,

	Bluriness,
	Threshold,
	ThresholdSmoothness,
	Exposure,
	Decay,
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
		Params::Echo,
		"Echo Amount",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(4.0);
			f.set_value(4.0);
			f.set_slider_min(2.0);
			f.set_slider_max(32.0);
			f.set_valid_min(2.0);
			f.set_valid_max(32.0);
			f.set_precision(Precision::Integer);
		}),
	)?;

	params.add_group(Params::TransformStart, Params::TransformEnd, "Transform", true, |params| {
		params.add(
			Params::TransformPosition,
			"Position",
			ae::PointDef::setup(|f| {
				f.set_default((0.0, 0.0));
				f.set_value(f.default());
			}),
		)?;

		params.add(
			Params::TransformRotation,
			"Rotation",
			ae::AngleDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(f.default());
			}),
		)?;

		params.add(
			Params::TransformScale,
			"Scale",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(110.0);
				f.set_value(f.default());
				f.set_slider_min(0.0);
				f.set_slider_max(300.0);
				f.set_valid_min(0.0);
				f.set_valid_max(300.0);
				f.set_display_flags(ValueDisplayFlag::PERCENT);
				f.set_precision(Precision::Hundredths);
			}),
		)?;

		Ok(())
	})?;

	params.add(
		Params::Bluriness,
		"Bluriness",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.0);
			f.set_value(f.default());
			f.set_slider_min(0.0);
			f.set_slider_max(500.0);
			f.set_valid_min(0.0);
			f.set_valid_max(500.0);
			f.set_display_flags(ValueDisplayFlag::PERCENT);
			f.set_precision(Precision::Hundredths);
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
			f.set_options(&["Add", "Screen", "Overlay", "Color Dodge", "Maximum"]);
		}),
	)?;

	params.add_group(Params::HelperStart, Params::HelperEnd, "Helper", true, |params| {
		params.add(
			Params::PreviewLayer,
			"Preview Output",
			ae::PopupDef::setup(|f| {
				f.set_default(1);
				f.set_options(&["Final Output", "Threshold Mask", "Echoes only"]);
			}),
		)?;

		Ok(())
	})?;

	Ok(())
}
