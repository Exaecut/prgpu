use after_effects::{self as ae, Error, InData, OutData, ParamFlag, Parameters};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum Params {
	Help,
	ReloadShaders,
	Feedback,

	LayersStart,

	Distortion,
	Compression,
	SignalNoise,
	Filter,

	LayersEnd,

	DistortionStart,

	DistortionAspectRatio,
	DistortionHorizontal,
	DistortionVertical,
	DistortionVignetteStrength,

	DistortionEnd,

	SNStart,

	SNTapeNoiseStart,

	SNLowfreqGlitch,
	SNHighfreqGlitch,
	SNHorizontalOffset,
	SNVerticalOffset,

	SNTapeNoiseEnd,

	SNTapeCreaseStart,

	SNTapeCreasePhaseFreq,
	SNTapeCreaseSpeed,
	SNTapeCreaseHeight,
	SNTapeCreaseDepth,
	SNTapeCreaseIntensity,
	SNTapeCreaseNoiseFreq,
	SNTapeCreaseStability,
	SNTapeCreaseMinimum,

	SNTapeCreaseEnd,

	SNExtremisNoiseHFrac,
	SNBorderLeakIntensity,
	SNBloomExposure,

	SNEnd,

	FilterStart,
	TintColor,
	PixelCellSize,
	ScanlineHardness,
	PixelHardness,
	BloomScanlineHardness,
	BloomPixelHardness,
	CRTContrast,
	FilterEnd,

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

	params.add_group(Params::LayersStart, Params::LayersEnd, "Layers", false, |params| {
		params.add(
			Params::Distortion,
			"Distortion",
			ae::CheckBoxDef::setup(|f| {
				f.set_default(true);
				f.set_value(true);
			}),
		)?;
		params.add(
			Params::Compression,
			"Compression",
			ae::CheckBoxDef::setup(|f| {
				f.set_default(true);
				f.set_value(true);
			}),
		)?;
		params.add(
			Params::SignalNoise,
			"Signal Noise",
			ae::CheckBoxDef::setup(|f| {
				f.set_default(true);
				f.set_value(true);
			}),
		)?;
		params.add(
			Params::Filter,
			"Filter",
			ae::CheckBoxDef::setup(|f| {
				f.set_default(true);
				f.set_value(true);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(Params::DistortionStart, Params::DistortionEnd, "Distortion Parameters", true, |params| {
		params.add(
			Params::DistortionAspectRatio,
			"Aspect Ratio",
			ae::PopupDef::setup(|f| {
				f.set_options(&["Use Original", "Fit to 4:3"]);
				f.set_default(0);
				f.set_value(0);
			}),
		)?;

		params.add(
			Params::DistortionHorizontal,
			"Horizontal Distortion",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.5);
				f.set_value(1.5);
				f.set_valid_min(-5.0);
				f.set_valid_max(5.0);
				f.set_slider_min(-5.0);
				f.set_slider_max(5.0);
				f.set_precision(ae::Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::DistortionVertical,
			"Vertical Distortion",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(1.5);
				f.set_value(1.5);
				f.set_valid_min(-5.0);
				f.set_valid_max(5.0);
				f.set_slider_min(-5.0);
				f.set_slider_max(5.0);
				f.set_precision(ae::Precision::Thousandths);
			}),
		)?;
		params.add(
			Params::DistortionVignetteStrength,
			"Vignette Strength",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.3);
				f.set_value(0.3);
				f.set_valid_min(0.0);
				f.set_valid_max(20.0);
				f.set_slider_min(0.0);
				f.set_slider_max(20.0);
				f.set_precision(ae::Precision::Thousandths);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(Params::SNStart, Params::SNEnd, "Signal Noise Parameters", true, |params| {
		params.add_group(Params::SNTapeNoiseStart, Params::SNTapeNoiseEnd, "Tape Noise", true, |params| {
			params.add(
				Params::SNLowfreqGlitch,
				"Low Frequency Glitch",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.005);
					f.set_value(0.005);
					f.set_valid_min(0.0);
					f.set_valid_max(0.2);
					f.set_slider_min(0.0);
					f.set_slider_max(0.2);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNHighfreqGlitch,
				"High Frequency Glitch",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.01);
					f.set_value(0.01);
					f.set_valid_min(0.0);
					f.set_valid_max(1.0);
					f.set_slider_min(0.0);
					f.set_slider_max(1.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNHorizontalOffset,
				"Horizontal Offset",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.5);
					f.set_value(0.5);
					f.set_valid_min(-3.0);
					f.set_valid_max(3.0);
					f.set_slider_min(-3.0);
					f.set_slider_max(3.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNVerticalOffset,
				"Vertical Offset",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.5);
					f.set_value(0.5);
					f.set_valid_min(-3.0);
					f.set_valid_max(3.0);
					f.set_slider_min(-3.0);
					f.set_slider_max(3.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			Ok(())
		})?;

		params.add_group(Params::SNTapeCreaseStart, Params::SNTapeCreaseEnd, "Tape Crease", true, |params| {
			params.add(
				Params::SNTapeCreasePhaseFreq,
				"Phase Frequency",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(9.0);
					f.set_value(9.0);
					f.set_valid_min(0.0);
					f.set_valid_max(50.0);
					f.set_slider_min(0.0);
					f.set_slider_max(50.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseSpeed,
				"Speed",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.15);
					f.set_value(0.15);
					f.set_valid_min(0.0);
					f.set_valid_max(10.0);
					f.set_slider_min(0.0);
					f.set_slider_max(10.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseHeight,
				"Height",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.2);
					f.set_value(0.2);
					f.set_valid_min(0.0);
					f.set_valid_max(1.0);
					f.set_slider_min(0.0);
					f.set_slider_max(1.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseDepth,
				"Depth",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.01);
					f.set_value(0.01);
					f.set_valid_min(0.0);
					f.set_valid_max(0.5);
					f.set_slider_min(0.0);
					f.set_slider_max(0.5);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseIntensity,
				"Intensity",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(10.0);
					f.set_value(10.0);
					f.set_valid_min(0.0);
					f.set_valid_max(30.0);
					f.set_slider_min(0.0);
					f.set_slider_max(30.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseNoiseFreq,
				"Noise Frequency",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(100.0);
					f.set_value(100.0);
					f.set_valid_min(0.0);
					f.set_valid_max(500.0);
					f.set_slider_min(0.0);
					f.set_slider_max(500.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseStability,
				"Stability",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.5);
					f.set_value(0.5);
					f.set_valid_min(0.0);
					f.set_valid_max(1.0);
					f.set_slider_min(0.0);
					f.set_slider_max(1.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			params.add(
				Params::SNTapeCreaseMinimum,
				"Minimum",
				ae::FloatSliderDef::setup(|f| {
					f.set_default(0.0);
					f.set_value(0.0);
					f.set_valid_min(0.0);
					f.set_valid_max(1.0);
					f.set_slider_min(0.0);
					f.set_slider_max(1.0);
					f.set_precision(ae::Precision::Thousandths);
				}),
			)?;

			Ok(())
		})?;

		params.add(
			Params::SNExtremisNoiseHFrac,
			"Extremis Noise Height",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.025);
				f.set_value(0.025);
				f.set_valid_min(0.0);
				f.set_valid_max(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(1.0);
				f.set_precision(ae::Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::SNBorderLeakIntensity,
			"Border Leak Intensity",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.2);
				f.set_value(0.2);
				f.set_valid_min(0.0);
				f.set_valid_max(2.0);
				f.set_slider_min(0.0);
				f.set_slider_max(2.0);
				f.set_precision(ae::Precision::Thousandths);
			}),
		)?;

		params.add(
			Params::SNBloomExposure,
			"Bloom Exposure",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.0625);
				f.set_value(0.0625);
				f.set_valid_min(0.0);
				f.set_valid_max(1.0);
				f.set_slider_min(0.0);
				f.set_slider_max(1.0);
				f.set_precision(ae::Precision::Thousandths);
			}),
		)?;

		Ok(())
	})?;

	params.add_group(Params::FilterStart, Params::FilterEnd, "Filters", true, |params| {
		params.add(
			Params::PixelCellSize,
			"Downscale Factor",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(2.0);
				f.set_value(2.0);
				f.set_valid_min(1.0);
				f.set_valid_max(10.0);
				f.set_slider_min(1.0);
				f.set_slider_max(10.0);
			}),
		)?;

		params.add(
			Params::ScanlineHardness,
			"Scanline Hardness",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(4.0);
				f.set_value(4.0);
				f.set_valid_min(0.0);
				f.set_valid_max(20.0);
				f.set_slider_min(0.0);
				f.set_slider_max(20.0);
				f.set_precision(ae::Precision::Tenths);
			}),
		)?;

		params.add(
			Params::PixelHardness,
			"Pixel Hardness",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(2.0);
				f.set_value(2.0);
				f.set_valid_min(0.0);
				f.set_valid_max(20.0);
				f.set_slider_min(0.0);
				f.set_slider_max(20.0);
				f.set_precision(ae::Precision::Tenths);
			}),
		)?;

		params.add(
			Params::BloomScanlineHardness,
			"Bloom Scanline Hardness",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(4.0);
				f.set_value(4.0);
				f.set_valid_min(0.0);
				f.set_valid_max(20.0);
				f.set_slider_min(0.0);
				f.set_slider_max(20.0);
				f.set_precision(ae::Precision::Tenths);
			}),
		)?;

		params.add(
			Params::BloomPixelHardness,
			"Bloom Pixel Hardness",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(0.5);
				f.set_value(0.5);
				f.set_valid_min(0.0);
				f.set_valid_max(20.0);
				f.set_slider_min(0.0);
				f.set_slider_max(20.0);
				f.set_precision(ae::Precision::Tenths);
			}),
		)?;

		params.add(
			Params::CRTContrast,
			"CRT Contrast",
			ae::FloatSliderDef::setup(|f| {
				f.set_default(3.0);
				f.set_value(3.0);
				f.set_valid_min(1.0);
				f.set_valid_max(10.0);
				f.set_slider_min(1.0);
				f.set_slider_max(10.0);
				f.set_precision(ae::Precision::Tenths);
			}),
		)?;

		Ok(())
	})?;

	params.add(
		Params::TintColor,
		"Tint Color",
		ae::ColorDef::setup(|f| {
			f.set_default(ae::pf::Pixel8 {
				red: 255,
				green: 255,
				blue: 255,
				alpha: 255,
			});
			f.set_value(f.default());
		}),
	)?;

	Ok(())
}
