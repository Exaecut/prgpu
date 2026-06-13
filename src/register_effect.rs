/// Declarative effect registration. Generates the AE `define_effect!` and
/// Premiere `define_gpu_filter!` boilerplate, plus the license wiring.
///
/// No-license form:
/// ```ignore
/// prgpu::register_effect!(Playground);
/// ```
///
/// With license:
/// ```ignore
/// prgpu::register_effect!(Mindglow, license = MindglowLicense);
/// ```
#[macro_export]
macro_rules! register_effect {
	($effect:ident) => {
		$crate::register_effect!($effect, license = $crate::effect::NoLicense);
	};
	($effect:ident, license = $license:ty) => {
		#[doc(hidden)]
		mod __prgpu_registration {
			use super::*;

			include!(::core::concat!(::core::env!("OUT_DIR"), "/prgpu_effect_meta.rs"));

			#[derive(Default)]
			struct Plugin($crate::adobe::ae::EffectAdapter<$effect, $license>);

			impl AdobePluginGlobal for Plugin {
				fn params_setup(
					&self,
					p: &mut ::after_effects::Parameters<<$effect as $crate::Effect>::Params>,
					i: ::after_effects::InData,
					o: ::after_effects::OutData,
				) -> ::core::result::Result<(), ::after_effects::Error> {
					self.0.params_setup(p, i, o)
				}

				fn handle_command(
					&mut self,
					command: ::after_effects::Command,
					in_data: ::after_effects::InData,
					out_data: ::after_effects::OutData,
					params: &mut ::after_effects::Parameters<<$effect as $crate::Effect>::Params>,
				) -> ::core::result::Result<(), ::after_effects::Error> {
					self.0.handle_command(command, in_data, out_data, params)
				}
			}

			::after_effects::define_effect!(Plugin, (), <$effect as $crate::Effect>::Params);

			type PremiereGpu = $crate::adobe::premiere::GpuFilterAdapter<$effect, $license>;
			::premiere::define_gpu_filter!(PremiereGpu);
		}
	};
}

