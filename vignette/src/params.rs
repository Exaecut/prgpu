use after_effects::{self as ae, sys::PF_Pixel, InData, OutData, ParamFlag, Parameters, Precision, ValueDisplayFlag};
use prgpu::params::SetupParams;

#[repr(usize)]
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	ReloadShaders,
	Feedback,

	Tint,
	DarkenStrength,
	BlurStrength,
	Anchor,
	ScaleX,
	ScaleY,
	Noise,
	NoiseSize,
	NoiseTimeOffset,

	DarkenMin,
	DarkenMax,

	BlurQuality,
	BlurRadius,
	BlurMin,
	BlurMax,

	Debug,
	NoLicense,
}

impl From<Params> for usize {
	fn from(p: Params) -> usize {
		p as usize
	}
}

impl SetupParams for Params {
	fn setup(params: &mut Parameters<Self>, _in_data: InData, _out_data: OutData) -> Result<(), after_effects::Error> {
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
			Params::Tint,
			"Tint Color",
			ae::ColorDef::setup(|f| {
				f.set_default(PF_Pixel {
					alpha: 255,
					red: 0,
					green: 0,
					blue: 0,
				});
				f.set_value(PF_Pixel {
					alpha: 255,
					red: 0,
					green: 0,
					blue: 0,
				});
			}),
		)?;

		params.add(
			Params::Anchor,
			"Anchor",
			ae::PointDef::setup(|f| {
				f.set_default((50.0, 50.0));
				f.set_value((50.0, 50.0));
			}),
		)?;

		params.add(
			Params::ScaleX,
			"Scale - X",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_value(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Hundredths);
			}),
		)?;

		params.add(
			Params::ScaleY,
			"Scale - Y",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_value(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Hundredths);
			}),
		)?;

		params.add(
			Params::Noise,
			"Noise - Intensity",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_valid_min(0.0);
				f.set_valid_max(100.0);
				f.set_display_flags(ValueDisplayFlag::PERCENT);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::NoiseSize,
			"Noise - Size",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_value(1.0);
				f.set_slider_min(1.0);
				f.set_slider_max(100.0);
				f.set_valid_min(1.0);
				f.set_valid_max(100.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::NoiseTimeOffset,
			"Noise - Phase",
			ae::AngleDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
			}),
		)?;

		params.add(
			Params::DarkenStrength,
			"Vignette Strength",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(100.0);
				f.set_value(100.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_valid_min(0.0);
				f.set_valid_max(100.0);
				f.set_display_flags(ValueDisplayFlag::PERCENT);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::DarkenMin,
			"Vignette - Mask Inner Radius",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
				f.set_slider_min(0.0);
				f.set_slider_max(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::DarkenMax,
			"Vignette - Mask Outer Radius",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_value(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::BlurStrength,
			"Blur Strength",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(10.0);
				f.set_value(10.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_valid_min(0.0);
				f.set_valid_max(100.0);
				f.set_precision(Precision::Thousandths);
				f.set_display_flags(ValueDisplayFlag::PERCENT);
			}),
		)?;

		params.add(
			Params::BlurQuality,
			"Blur - Quality",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(2.0);
				f.set_value(2.0);
				f.set_slider_min(1.0);
				f.set_slider_max(32.0);
				f.set_valid_min(1.0);
				f.set_valid_max(32.0);
				f.set_precision(Precision::Integer);
			}),
		)?;

		params.add(
			Params::BlurRadius,
			"Blur - Radius",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_value(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::BlurMin,
			"Blur - Mask Inner Radius",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0);
				f.set_value(0.0);
				f.set_slider_min(0.0);
				f.set_slider_max(128.0);
				f.set_valid_min(0.0);
				f.set_valid_max(128.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::BlurMax,
			"Blur - Mask Outer Radius",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_value(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		Ok(())
	}
}
