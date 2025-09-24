use after_effects::{self as ae, pf, Error, InData, OutData, ParamFlag, Parameters, Precision};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,
	Clip,
	Mode,
	Steps,
	Spread,
	Angle,

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

		if cfg!(shader_hotreload) {
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
	}

	params.add(
		Params::Mode,
		"Mode",
		ae::PopupDef::setup(|f| {
			f.set_options(&["Linear", "Radial"]);
			f.set_default(1);
			f.set_value(1);
		}),
	)?;

	params.add(
		Params::Clip,
		"Clip at bounds",
		ae::CheckBoxDef::setup(|f| {
			f.set_label("Clip");
			f.set_default(true);
			f.set_value(true);
		}),
	)?;

	params.add(
		Params::Steps,
		"Steps",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(2.0);
			f.set_slider_min(2.0);
			f.set_slider_max(256.0);
			f.set_valid_min(2.0);
			f.set_valid_max(256.0);
			f.set_precision(Precision::Integer);
		}),
	)?;

	params.add(
		Params::Spread,
		"Spread",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(1.25);
			f.set_slider_min(0.0);
			f.set_slider_max(200.0);
			f.set_valid_min(0.0);
			f.set_valid_max(200.0);
			f.set_precision(Precision::Hundredths);
			f.set_display_flags(pf::ValueDisplayFlag::PERCENT);
		}),
	)?;

	params.add(
		Params::Angle,
		"Angle",
		ae::AngleDef::setup(|f| {
			f.set_default(15.0);
			f.set_value(15.0);
		}),
	)?;

	Ok(())
}
