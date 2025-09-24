use after_effects::{self as ae, Error, InData, OutData, ParamFlag, Parameters};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,
	Shift,

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
		Params::Shift,
		"Shift",
		ae::AngleDef::setup(|f| {
			f.set_default(0.0);
			f.set_value(0.0);
		}),
	)?;

	Ok(())
}
