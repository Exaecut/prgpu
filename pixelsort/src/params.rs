use after_effects::{self as ae, Error, InData, OutData, ParamFlag, Parameters};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,

	DarkLuminanceThreshold,
	LightLuminanceThreshold,
	IncrementDivisor,
	RandomIncrementRange,

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
		Params::DarkLuminanceThreshold,
		"Dark Luminance Threshold",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.0);
			f.set_valid_min(0.0);
			f.set_valid_max(100.0);
			f.set_slider_min(0.0);
			f.set_slider_max(100.0);
			f.set_display_flags(ae::ValueDisplayFlag::PERCENT);
		}),
	)?;

	params.add(Params::LightLuminanceThreshold, "Light Luminance Threshold", ae::FloatSliderDef::setup(|f| {
		f.set_default(50.0);
		f.set_valid_min(0.0);
		f.set_valid_max(100.0);
		f.set_slider_min(0.0);
		f.set_slider_max(100.0);
		f.set_display_flags(ae::ValueDisplayFlag::PERCENT);
	}))?;

	params.add(Params::IncrementDivisor, "Increment Divisor", ae::FloatSliderDef::setup(|f| {
		f.set_default(30.0);
		f.set_valid_min(0.0);
		f.set_valid_max(100.0);
		f.set_slider_min(0.0);
		f.set_slider_max(100.0);
	}))?;

	params.add(Params::RandomIncrementRange, "Random Increment Range", ae::FloatSliderDef::setup(|f| {
		f.set_default(6.0);
		f.set_valid_min(0.0);
		f.set_valid_max(100.0);
		f.set_slider_min(0.0);
		f.set_slider_max(100.0);
	}))?;

	params.add_group(Params::HelperStart, Params::HelperEnd, "Helper", true, |params| {
		params.add(
			Params::PreviewLayer,
			"Preview Output",
			ae::PopupDef::setup(|f| {
				f.set_default(0);
				f.set_options(&["Final Output", "Threshold Input"]);
			}),
		)?;

		Ok(())
	})?;

	Ok(())
}
