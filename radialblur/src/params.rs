use after_effects::{self as ae, log, Error, InData, OutData, ParamFlag, Parameters};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,
	Debug,
	Type,
	Angle,
	InnerSpread,
	OuterSpread,
	SpreadFade,
	OffsetsStart,
	RedOffset,
	GreenOffset,
	BlueOffset,
	OffsetsEnd,
	Samples,
	Origin,
	BlurAlpha,
	UniformAspectRatio,
	NoLicense,
}

pub fn setup(params: &mut Parameters<Params>, _in_data: InData, _out_data: OutData) -> Result<(), Error> {
	log::info!("Setting up parameters");
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
		Params::Type,
		"Type",
		ae::PopupDef::setup(|f| {
			f.set_options(&["Angular", "Linear"]);
			f.set_default(0);
			f.set_value(0);
		}),
	)?;

	params.add(
		Params::Angle,
		"Angle / Distance",
		ae::AngleDef::setup(|f| {
			f.set_default(15.0);
			f.set_value(15.0);
		}),
	)?;

	params.add(
		Params::InnerSpread,
		"Inner Spread",
		ae::FloatSliderDef::setup(|f| {
			f.set_slider_min(0.0);
			f.set_slider_max(1.0);
			f.set_valid_min(0.0);
			f.set_valid_max(1.0);
			f.set_default(0.0);
			f.set_precision(ae::Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::OuterSpread,
		"Outer Spread",
		ae::FloatSliderDef::setup(|f| {
			f.set_slider_min(0.0);
			f.set_slider_max(5.0);
			f.set_valid_min(0.0);
			f.set_valid_max(5.0);
			f.set_default(1.0);
			f.set_precision(ae::Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::SpreadFade,
		"Spread Fade",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(50.0);
			f.set_slider_min(0.0);
			f.set_slider_max(200.0);
			f.set_valid_min(0.0);
			f.set_valid_max(200.0);
			f.set_precision(ae::Precision::Hundredths);
		}),
	)?;

	params.add(
		Params::Samples,
		"Samples",
		ae::FloatSliderDef::setup(|f| {
			f.set_slider_min(2.0);
			f.set_slider_max(1024.0);
			f.set_valid_min(2.0);
			f.set_valid_max(1024.0);
			f.set_default(128.0);
			f.set_precision(ae::Precision::Integer);
		}),
	)?;

	params.add(
		Params::Origin,
		"Origin",
		ae::PointDef::setup(|f| {
			f.set_default((50.0, 50.0));
		}),
	)?;

	params.add_group(Params::OffsetsStart, Params::OffsetsEnd, "Chromatic Aberration", false, |params| {
		params.add(
			Params::RedOffset,
			"Red Offset",
			ae::AngleDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
			}),
		)?;

		params.add(
			Params::GreenOffset,
			"Green Offset",
			ae::AngleDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
			}),
		)?;

		params.add(
			Params::BlueOffset,
			"Blue Offset",
			ae::AngleDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
			}),
		)?;

		Ok(())
	})?;

	params.add(
		Params::UniformAspectRatio,
		"Uniform Blur Aspect Ratio",
		ae::CheckBoxDef::setup(|f| {
			f.set_label("Enable");
			f.set_default(false);
			f.set_value(false);
		}),
	)?;

	params.add(
		Params::BlurAlpha,
		"Blur Alpha",
		ae::CheckBoxDef::setup(|f| {
			f.set_label("Enable");
			f.set_default(false);
			f.set_value(false);
		}),
	)?;

	Ok(())
}
