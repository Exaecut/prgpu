use after_effects::{self as ae, sys::PF_Pixel, Error, InData, OutData, ParamFlag, Parameters, Precision};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,

	PlasmaMode,
	Speed,
	TimeOffset,

	Start(PlasmaModes),
	End(PlasmaModes),

	TDDepth,
	TDFractalRep,
	TDTwistFreq,
	TDTwistAmp,
	TDColorsStart,
	TDColor1,
	TDColor2,
	TDColor3,
	TDColor4,
	TDColorsEnd,

	EFPatDeter,
	EFFrequency,
	EFColor,

	TSIterations,
	TSWarpBase,

	SVColor,
	SVHeight,
	SVHeightOffset,
	SVVertScale,
	SVVertOffset,
	SVHorizontalScale,
	SVHorizontalOffset,
	SVHorizontalAngle,
	SVIterations,
	SVAccumulation,
	SVFogDensity,

	Debug,
	NoLicense,
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum PlasmaModes {
	TurbulentDepth,
	EnergyFlux,
	TheSun,
	SweetVelvet,
}

impl PlasmaModes {
	pub fn from_u32(i: u32) -> Self {
		match i {
			1 => Self::TurbulentDepth,
			2 => Self::EnergyFlux,
			3 => Self::TheSun,
			4 => Self::SweetVelvet,
			_ => Self::TurbulentDepth,
		}
	}
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
		Params::PlasmaMode,
		"Plasma Type",
		ae::PopupDef::setup(|f| {
			f.set_default(1);
			f.set_options(&["Turbulent Depth", "Energy Flux", "The Sun !!!", "Sweet Velvet"]);
		}),
	)?;

	params.add(
		Params::Speed,
		"Speed",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(1.0);
			f.set_valid_min(0.01);
			f.set_valid_max(10.0);
			f.set_slider_min(0.01);
			f.set_slider_max(10.0);
			f.set_precision(Precision::Thousandths);
		}),
	)?;

	params.add(
		Params::TimeOffset,
		"Time Offset",
		ae::FloatSliderDef::setup(|f| {
			f.set_default(0.0);
			f.set_valid_min(0.0);
			f.set_valid_max(10000.0);
			f.set_slider_min(0.0);
			f.set_slider_max(10000.0);
			f.set_precision(Precision::Hundredths);
		}),
	)?;

	params.add_group(
		Params::Start(PlasmaModes::SweetVelvet),
		Params::End(PlasmaModes::SweetVelvet),
		"Sweet Velvet",
		false,
		|params| {
			params.add(
				Params::SVColor,
				"Color",
				ae::ColorDef::setup(|f| {
					f.set_default(PF_Pixel {
						alpha: 255,
						red: 100,
						green: 0,
						blue: 255,
					});
				}),
			)?;

			params.add(
				Params::SVHeight,
				"Height",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(1.5);
					f.set_valid_min(0.01);
					f.set_valid_max(10.0);
					f.set_slider_min(0.01);
					f.set_slider_max(10.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SVHeightOffset,
				"Height Offset",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(5.0);
					f.set_valid_min(0.0);
					f.set_valid_max(100.0);
					f.set_slider_min(0.0);
					f.set_slider_max(100.0);
					f.set_precision(Precision::Hundredths);
				}),
			)?;

			params.add(Params::SVVertScale, "Vertical Movement Stability", ae::FloatSliderDef::setup(|f| {
				f.set_default(3.0);
				f.set_valid_min(0.1);
				f.set_valid_max(10.0);
				f.set_slider_min(0.1);
				f.set_slider_max(10.0);
				f.set_precision(Precision::Thousandths);
			}))?;

			params.add(Params::SVVertOffset, "Vertical Offset", ae::FloatSliderDef::setup(|f| {
				f.set_default(1.5);
				f.set_valid_min(0.0);
				f.set_valid_max(10.0);
				f.set_slider_min(0.0);
				f.set_slider_max(10.0);
				f.set_precision(Precision::Thousandths);
			}))?;

			params.add(Params::SVHorizontalScale, "Horizontal Movement Stability", ae::FloatSliderDef::setup(|f| {
				f.set_default(2.0);
				f.set_valid_min(0.1);
				f.set_valid_max(10.0);
				f.set_slider_min(0.1);
				f.set_slider_max(10.0);
				f.set_precision(Precision::Thousandths);
			}))?;

			params.add(Params::SVHorizontalOffset, "Horizontal Offset", ae::FloatSliderDef::setup(|f| {
				f.set_default(1.0);
				f.set_valid_min(0.0);
				f.set_valid_max(10.0);
				f.set_slider_min(0.0);
				f.set_slider_max(10.0);
				f.set_precision(Precision::Thousandths);	
			}))?;

			params.add(Params::SVHorizontalAngle, "Horizontal Angular Scale", ae::FloatSliderDef::setup(|f| {
				f.set_default(5.0);
				f.set_valid_min(0.0);
				f.set_valid_max(100.0);
				f.set_slider_min(0.0);
				f.set_slider_max(100.0);
				f.set_precision(Precision::Thousandths);
			}))?;

			params.add(Params::SVIterations, "Iterations", ae::SliderDef::setup(|f| {
				f.set_default(40);
				f.set_valid_min(1);
				f.set_valid_max(250);
				f.set_slider_min(1);
				f.set_slider_max(250);
			}))?;

			params.add(Params::SVAccumulation, "Accumulation", ae::FloatSliderDef::setup(|f| {
				f.set_default(0.5);
				f.set_valid_min(0.0);
				f.set_valid_max(3.0);
				f.set_slider_min(0.0);
				f.set_slider_max(3.0);
				f.set_precision(Precision::Thousandths);
			}))?;

			params.add(Params::SVFogDensity, "Fog Density", ae::FloatSliderDef::setup(|f| {
				f.set_default(0.005);
				f.set_valid_min(0.0);
				f.set_valid_max(0.5);
				f.set_slider_min(0.0);
				f.set_slider_max(0.5);
				f.set_precision(Precision::Thousandths);
			}))?;

			Ok(())
		},
	)?;

	params.add_group(Params::Start(PlasmaModes::TheSun), Params::End(PlasmaModes::TheSun), "The Sun", false, |params| {
		params.add(
			Params::TSIterations,
			"Iterations",
			ae::SliderDef::setup(|f| {
				f.set_default(80);
				f.set_valid_min(1);
				f.set_valid_max(250);
				f.set_slider_min(1);
				f.set_slider_max(250);
			}),
		)?;

		params.add(
			Params::TSWarpBase,
			"Warp Base",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.012);
				f.set_valid_min(0.0);
				f.set_valid_max(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(1.0);
				f.set_precision(Precision::Thousandths);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(
		Params::Start(PlasmaModes::EnergyFlux),
		Params::End(PlasmaModes::EnergyFlux),
		"Energy Flux",
		false,
		|params| {
			params.add(
				Params::EFPatDeter,
				"Pattern determinism",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(1.5);
					f.set_valid_min(0.01);
					f.set_valid_max(10.0);
					f.set_slider_min(0.01);
					f.set_slider_max(10.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::EFFrequency,
				"Frequency",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(1.0);
					f.set_valid_min(0.01);
					f.set_valid_max(10.0);
					f.set_slider_min(0.01);
					f.set_slider_max(10.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::EFColor,
				"Color",
				ae::ColorDef::setup(|f| {
					f.set_default(PF_Pixel {
						red: 20,
						green: 255,
						blue: 60,
						alpha: 255,
					});
				}),
			)?;

			Ok(())
		},
	)?;

	params.add_group(
		Params::Start(PlasmaModes::TurbulentDepth),
		Params::End(PlasmaModes::TurbulentDepth),
		"Turbulent Depth",
		false,
		|params| {
			params.add(
				Params::TDDepth,
				"Depth",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(100.0);
					f.set_valid_min(1.00);
					f.set_valid_max(200.0);
					f.set_slider_min(1.00);
					f.set_slider_max(200.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::TDFractalRep,
				"Fractal Repetition",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(40.0);
					f.set_valid_min(0.01);
					f.set_valid_max(300.0);
					f.set_slider_min(0.01);
					f.set_slider_max(300.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::TDTwistFreq,
				"Twist Frequency",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.35);
					f.set_valid_min(0.01);
					f.set_valid_max(10.0);
					f.set_slider_min(0.01);
					f.set_slider_max(10.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::TDTwistAmp,
				"Twist Amplitude",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(1.2);
					f.set_valid_min(0.01);
					f.set_valid_max(10.0);
					f.set_slider_min(0.01);
					f.set_slider_max(10.0);
					f.set_precision(Precision::Thousandths);
				}),
			)?;

			params.add_group(Params::TDColorsStart, Params::TDColorsEnd, "Colors", true, |params| {
				params.add(
					Params::TDColor1,
					"Color 1",
					ae::ColorDef::setup(|f| {
						f.set_default(PF_Pixel {
							red: 255,
							green: 50,
							blue: 80,
							alpha: 255,
						});
					}),
				)?;

				params.add(
					Params::TDColor2,
					"Color 2",
					ae::ColorDef::setup(|f| {
						f.set_default(PF_Pixel {
							red: 80,
							green: 255,
							blue: 50,
							alpha: 255,
						});
					}),
				)?;

				params.add(
					Params::TDColor3,
					"Color 3",
					ae::ColorDef::setup(|f| {
						f.set_default(PF_Pixel {
							red: 50,
							green: 80,
							blue: 255,
							alpha: 255,
						});
					}),
				)?;

				params.add(
					Params::TDColor4,
					"Color 4",
					ae::ColorDef::setup(|f| {
						f.set_default(PF_Pixel {
							red: 200,
							green: 50,
							blue: 255,
							alpha: 255,
						});
					}),
				)?;

				Ok(())
			})?;

			Ok(())
		},
	)?;

	Ok(())
}
