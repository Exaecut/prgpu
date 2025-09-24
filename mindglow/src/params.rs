use after_effects::{self as ae, pf, Error, InData, OutData, ParamFlag, Parameters, Precision};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,
	AllowOOBGlow,
	Strength,
	LayerCount,
	LayerSizesStart,
	LayerSizeIndex(usize),
	LayerSizesEnd,
	Radius,
	TintColor,

	InputStart,
	Threshold,
	ThresholdSmoothness,
	InputEnd,

	StyleStart,

	EffectStart,
	ChromaticAberration,
	Flicker,
	FlickerFrequency,
	FlickerRandomness,
	FlickerBias,
	EffectEnd,

	StyleEnd,

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
		Params::Strength,
		"Strength",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(1.0);
			f.set_slider_min(0.0);
			f.set_slider_max(10.0);
			f.set_valid_min(0.0);
			f.set_valid_max(1000.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add_with_flags(
		Params::LayerCount,
		"Layer count",
		ae::SliderDef::setup(|f| {
			f.set_default(8);
			f.set_slider_min(2);
			f.set_slider_max(10);
			f.set_valid_min(2);
			f.set_valid_max(10);
		}),
		ae::ParamFlag::SUPERVISE,
		ae::ParamUIFlags::empty(),
	)?;

	params.add(
		Params::AllowOOBGlow,
		"Allow out-of-bounds glow",
		ae::CheckBoxDef::setup(|f| {
			f.set_default(false);
		}),
	)?;

	params.add_group(Params::LayerSizesStart, Params::LayerSizesEnd, "Per-layer brightness", true, |params| {
		let default_values: [f64; 10] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

		for (i, &default_value) in default_values.iter().enumerate() {
			params.add(
				Params::LayerSizeIndex(i),
				format!("Layer {}", i + 1).as_str(),
				ae::FloatSliderDef::setup(|f| {
					f.set_default(default_value);
					f.set_slider_min(0.0);
					f.set_slider_max(10.0);
					f.set_valid_min(0.0);
					f.set_valid_max(10.0);
					f.set_precision(Precision::Hundredths);
				}),
			)?;
		}
		Ok(())
	})?;

	params.add(
		Params::Radius,
		"Radius",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.05);
			f.set_slider_min(0.05);
			f.set_slider_max(512.0);
			f.set_valid_min(0.05);
			f.set_valid_max(512.0);
			f.set_precision(Precision::Thousandths);
		}),
	)?;

	params.add(
		Params::TintColor,
		"Tint Color",
		ae::ColorDef::setup(|f| {
			f.set_default(pf::Pixel8 {
				red: 255,
				green: 255,
				blue: 255,
				alpha: 255,
			});
			f.set_value(f.default());
		}),
	)?;

	params.add_group(Params::InputStart, Params::InputEnd, "Input", true, |params| {
		params.add(
			Params::Threshold,
			"Threshold",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0);
				f.set_slider_min(0.0);
				f.set_slider_max(1.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::ThresholdSmoothness,
			"Threshold Smoothness",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0);
				f.set_slider_min(0.0);
				f.set_slider_max(1.0);
				f.set_valid_min(0.0);
				f.set_valid_max(5.0);
				f.set_precision(Precision::Hundredths);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(Params::StyleStart, Params::StyleEnd, "Style", true, |params| {
		params.add_group(Params::EffectStart, Params::EffectEnd, "Effect", true, |params| {
			params.add(
				Params::ChromaticAberration,
				"Chromatic Aberration",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.0);
					f.set_slider_min(0.0);
					f.set_slider_max(1.0);
					f.set_valid_min(0.0);
					f.set_valid_max(1.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::Flicker,
				"Flicker",
				ae::CheckBoxDef::setup(|f| {
					f.set_label("Flicker");
					f.set_default(false);
					f.set_value(false);
				}),
			)?;

			params.add(
				Params::FlickerFrequency,
				"Flicker Frequency",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(1.0);
					f.set_slider_min(0.0);
					f.set_slider_max(10.0);
					f.set_valid_min(0.0);
					f.set_valid_max(10.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::FlickerRandomness,
				"Flicker Randomness",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.2);
					f.set_slider_min(0.0);
					f.set_slider_max(1.0);
					f.set_valid_min(0.0);
					f.set_valid_max(1.0);
					f.set_precision(Precision::Hundredths);
				}),
			)?;

			params.add(
				Params::FlickerBias,
				"Flicker Bias",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.0);
					f.set_slider_min(0.0);
					f.set_slider_max(100.0);
					f.set_valid_min(0.0);
					f.set_valid_max(100.0);
					f.set_precision(Precision::Hundredths);
				}),
			)?;

			Ok(())
		})?;

		Ok(())
	})?;

	params.add_group(Params::HelperStart, Params::HelperEnd, "Helper", true, |params| {
		params.add(
			Params::PreviewLayer,
			"Preview Output",
			ae::PopupDef::setup(|f| {
				f.set_default(0);
				f.set_options(&["Final Output", "Bloom Only", "Alpha luminance"]);
			}),
		)?;

		Ok(())
	})?;

	Ok(())
}
