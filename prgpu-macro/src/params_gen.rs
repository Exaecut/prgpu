//! Codegen for `params!`: the discriminant enum, one marker per param, the
//! `ParamsSpec` (host registration + per-frame `Snapshot`), the `Snapshot`
//! storage, and the transitional `SetupParams` bridge.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::params_parse::{Kind, Node, ParamDef, ParamsInput, PopupSource, PopupTy, marker_end, marker_start};

pub fn generate(input: ParamsInput) -> TokenStream {
	let ParamsInput {
		vis,
		enum_ident,
		variants,
		params,
		nodes,
	} = input;

	let n = variants.len();
	let count = lit_usize(n);

	let first = &variants[0];
	let rest = &variants[1..];
	let enum_def = quote! {
		#[repr(usize)]
		#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
		#vis enum #enum_ident {
			#first = 1,
			#( #rest, )*
		}

		impl ::core::convert::From<#enum_ident> for usize {
			fn from(p: #enum_ident) -> usize {
				p as usize
			}
		}
	};

	// Declaration-order index of each layer param, used both for the marker's
	// inherent `LAYER_INDEX` const (pipelines bind `Slot::Layer(LAYER_INDEX)`)
	// and to slot the param into `InvocationBase::layers` at checkout.
	let mut layer_ordinal = 0u32;
	let markers = params.iter().map(|p| {
		let id = &p.ident;
		let vty = value_ty(&p.kind);
		let layer_const = if matches!(p.kind, Kind::Layer { .. }) {
			let idx = lit_u32(layer_ordinal);
			layer_ordinal += 1;
			quote! {
				impl #id {
					/// Secondary-input slot index for `Slot::Layer(..)` and
					/// `Ctx::layer_present(..)`.
					pub const LAYER_INDEX: u32 = #idx;
				}
			}
		} else {
			quote! {}
		};
		quote! {
			#[derive(Clone, Copy)]
			#vis struct #id;
			impl ::prgpu::Param for #id {
				type Spec = #enum_ident;
				type Value = #vty;
				const ID: #enum_ident = #enum_ident::#id;
			}
			#layer_const
		}
	}).collect::<Vec<_>>();

	// Layer params in declaration order, so the adapter checks each out into the
	// matching `InvocationBase::layers` slot.
	let layer_params: Vec<&Ident> = params.iter().filter(|p| matches!(p.kind, Kind::Layer { .. })).map(|p| &p.ident).collect();
	let layer_params_tokens = quote! { &[ #( #enum_ident::#layer_params ),* ] };

	let snapshot = quote! {
		#[doc(hidden)]
		#[derive(Clone, Copy)]
		#vis struct __Snapshot([::prgpu::ParamValue; #count]);

		impl ::core::default::Default for __Snapshot {
			fn default() -> Self {
				Self([::prgpu::ParamValue::None; #count])
			}
		}

		impl ::prgpu::Snapshot<#enum_ident> for __Snapshot {
			fn value(&self, id: #enum_ident) -> ::prgpu::ParamValue {
				self.0[(id as usize) - 1]
			}
			fn set(&mut self, id: #enum_ident, value: ::prgpu::ParamValue) {
				self.0[(id as usize) - 1] = value;
			}
		}
	};

	let register_stmts: Vec<TokenStream> = nodes.iter().map(|node| emit_node(node, &enum_ident, &params)).collect();
	let cpu_stmts: Vec<TokenStream> = params.iter().filter_map(|p| cpu_stmt(p, &enum_ident)).collect();
	let gpu_stmts: Vec<TokenStream> = params.iter().filter_map(|p| gpu_stmt(p, &enum_ident)).collect();

	let buttons = buttons(&params, &enum_ident);
	let debug_param = debug_param(&params, &enum_ident);

	quote! {
		#enum_def

		#( #markers )*

		#snapshot

		impl ::prgpu::ParamsSpec for #enum_ident {
			const COUNT: usize = #count;
			const DEBUG_PARAM: ::core::option::Option<Self> = #debug_param;
			const LAYER_PARAMS: &'static [Self] = #layer_params_tokens;
			type Snapshot = __Snapshot;

			#[allow(unused_variables)]
			fn register(params: &mut ::after_effects::Parameters<Self>) -> ::core::result::Result<(), ::after_effects::Error> {
				#( #register_stmts )*
				Ok(())
			}

			#[allow(unused_variables)]
			fn snapshot_cpu(params: &::after_effects::Parameters<Self>, geom: &::prgpu::SnapshotGeom) -> ::core::result::Result<__Snapshot, ::after_effects::Error> {
				let mut snapshot = __Snapshot::default();
				#( #cpu_stmts )*
				Ok(snapshot)
			}

			#[allow(unused_variables)]
			fn snapshot_gpu(filter: &::premiere::GpuFilterData, render_params: &::premiere::RenderParams, geom: &::prgpu::SnapshotGeom) -> __Snapshot {
				let mut snapshot = __Snapshot::default();
				#( #gpu_stmts )*
				snapshot
			}

			fn buttons() -> &'static [(Self, fn())] {
				#buttons
			}
		}

		// TRANSITIONAL(plan-05): legacy setup bridge so the old `Effect` path
		// keeps calling `Params::setup` until the trait/graph swap.
		impl ::prgpu::params::SetupParams for #enum_ident {
			fn setup(params: &mut ::after_effects::Parameters<Self>, _in_data: ::after_effects::InData, _out_data: ::after_effects::OutData) -> ::core::result::Result<(), ::after_effects::Error> {
				<Self as ::prgpu::ParamsSpec>::register(params)
			}
		}
	}
}

fn emit_node(node: &Node, enum_ident: &Ident, params: &[ParamDef]) -> TokenStream {
	match node {
		Node::Param(i) => reg_stmt(&params[*i], enum_ident),
		Node::Group { idx, name, collapsed, children } => {
			let start = marker_start(*idx);
			let end = marker_end(*idx);
			let kids = children.iter().map(|c| emit_node(c, enum_ident, params));
			quote! {
				params.add_group(#enum_ident::#start, #enum_ident::#end, #name, #collapsed, |params| {
					#( #kids )*
					Ok(())
				})?;
			}
		}
	}
}

fn reg_stmt(p: &ParamDef, enum_ident: &Ident) -> TokenStream {
	let id = &p.ident;
	let label = &p.label;
	match &p.kind {
		Kind::Slider {
			vmin,
			vmax,
			default,
			smin,
			smax,
			percent,
			precision,
		} => {
			let d = flit64(*default);
			let smn = flit32(*smin);
			let smx = flit32(*smax);
			let vmn = flit32(*vmin);
			let vmx = flit32(*vmax);
			let percent = if *percent {
				quote! { f.set_display_flags(::after_effects::ValueDisplayFlag::PERCENT); }
			} else {
				quote! {}
			};
			let prec = match precision {
				Some(p) => {
					let pl = i16lit(*p);
					quote! { f.set_precision(#pl); }
				}
				None => quote! {},
			};
			quote! {
				params.add(#enum_ident::#id, #label, ::after_effects::FloatSliderDef::setup(|f| {
					f.set_default(#d);
					f.set_value(#d);
					f.set_slider_min(#smn);
					f.set_slider_max(#smx);
					f.set_valid_min(#vmn);
					f.set_valid_max(#vmx);
					#percent
					#prec
				}))?;
			}
		}
		Kind::Checkbox { default, supervise } => {
			let d = if *default { quote! { true } } else { quote! { false } };
			let def = quote! {
				::after_effects::CheckBoxDef::setup(|f| {
					f.set_label(#label);
					f.set_default(#d);
					f.set_value(#d);
				})
			};
			let supervise_stmt = if *supervise {
				quote! { p.set_flag(::after_effects::ParamFlag::SUPERVISE, true); }
			} else {
				quote! {}
			};
			if p.debug_only {
				quote! {
					params.add_customized(#enum_ident::#id, #label, #def, |p| {
						#supervise_stmt
						if !cfg!(debug_assertions) {
							p.set_flag(::after_effects::ParamFlag::START_COLLAPSED, true);
							p.set_ui_flag(::after_effects::ParamUIFlags::INVISIBLE, true);
						}
						-1
					})?;
				}
			} else if *supervise {
				quote! {
					params.add_with_flags(#enum_ident::#id, #label, #def, ::after_effects::ParamFlag::SUPERVISE, ::after_effects::ParamUIFlags::empty())?;
				}
			} else {
				quote! { params.add(#enum_ident::#id, #label, #def)?; }
			}
		}
		Kind::Color { r, g, b, a } => {
			let (r, g, b, a) = (u8lit(*r), u8lit(*g), u8lit(*b), u8lit(*a));
			quote! {
				params.add(#enum_ident::#id, #label, ::after_effects::ColorDef::setup(|f| {
					f.set_default(::after_effects::sys::PF_Pixel { alpha: #a, red: #r, green: #g, blue: #b });
					f.set_value(f.default());
				}))?;
			}
		}
		Kind::Angle { default } => {
			let d = flit32(*default);
			quote! {
				params.add(#enum_ident::#id, #label, ::after_effects::AngleDef::setup(|f| {
					f.set_default(#d);
					f.set_value(#d);
				}))?;
			}
		}
		Kind::Point { x, y } => {
			let (x, y) = (flit32(*x), flit32(*y));
			quote! {
				params.add(#enum_ident::#id, #label, ::after_effects::PointDef::setup(|f| {
					f.set_default((#x, #y));
					f.set_value((#x, #y));
				}))?;
			}
		}
		Kind::Popup { options, default, .. } => {
			let options = match options {
				PopupSource::Inline(labels) => quote! { &[ #(#labels),* ] },
				PopupSource::Enum(path) => quote! { <#path as ::prgpu::PopupOptions>::LABELS },
			};
			// User-visible popup indices are 0-based; AE PF stores 1-based.
			let default1 = quote! { ((#default) as i32) + 1 };
			let def = quote! {
				::after_effects::PopupDef::setup(|f| {
					f.set_options(#options);
					f.set_default(#default1);
					f.set_value(#default1);
				})
			};
			if p.debug_only {
				quote! {
					params.add_customized(#enum_ident::#id, #label, #def, |p| {
						p.set_flag(::after_effects::ParamFlag::SUPERVISE, true);
						if !cfg!(debug_assertions) {
							p.set_flag(::after_effects::ParamFlag::START_COLLAPSED, true);
							p.set_ui_flag(::after_effects::ParamUIFlags::INVISIBLE, true);
						}
						-1
					})?;
				}
			} else {
				quote! {
					params.add_with_flags(#enum_ident::#id, #label, #def, ::after_effects::ParamFlag::SUPERVISE, ::after_effects::ParamUIFlags::empty())?;
				}
			}
		}
		Kind::Button { .. } => {
			quote! {
				params.add_customized(#enum_ident::#id, #label, ::after_effects::ButtonDef::setup(|f| {
					f.set_label(#label);
				}), |p| {
					p.set_flag(::after_effects::ParamFlag::SUPERVISE, true);
					p.set_flag(::after_effects::ParamFlag::START_COLLAPSED, true);
					-1
				})?;
			}
		}
		Kind::Layer { default_myself } => {
			let default_stmt = if *default_myself {
				quote! { f.set_default_to_this_layer(); }
			} else {
				quote! {}
			};
			quote! {
				params.add(#enum_ident::#id, #label, ::after_effects::LayerDef::setup(|f| {
					#default_stmt
				}))?;
			}
		}
		Kind::Custom { setup } => {
			quote! { (#setup)(params, #enum_ident::#id)?; }
		}
	}
}

fn cpu_stmt(p: &ParamDef, enum_ident: &Ident) -> Option<TokenStream> {
	let id = &p.ident;
	let call = match &p.kind {
		Kind::Slider { .. } => quote! { ::prgpu::params::convert::cpu_float(params, #enum_ident::#id)? },
		Kind::Angle { .. } => quote! { ::prgpu::params::convert::cpu_angle(params, #enum_ident::#id)? },
		Kind::Checkbox { .. } => quote! { ::prgpu::params::convert::cpu_checkbox(params, #enum_ident::#id)? },
		Kind::Color { .. } => quote! { ::prgpu::params::convert::cpu_color(params, #enum_ident::#id)? },
		Kind::Point { .. } => quote! { ::prgpu::params::convert::cpu_point(params, #enum_ident::#id, geom.layer_w, geom.layer_h)? },
		Kind::Popup { .. } => quote! { ::prgpu::params::convert::cpu_popup(params, #enum_ident::#id)? },
		Kind::Button { .. } | Kind::Layer { .. } | Kind::Custom { .. } => return None,
	};
	Some(quote! { ::prgpu::Snapshot::set(&mut snapshot, #enum_ident::#id, #call); })
}

fn gpu_stmt(p: &ParamDef, enum_ident: &Ident) -> Option<TokenStream> {
	let id = &p.ident;
	let call = match &p.kind {
		Kind::Slider { .. } | Kind::Angle { .. } => quote! { ::prgpu::params::convert::gpu_float(filter, render_params, #enum_ident::#id) },
		Kind::Checkbox { .. } => quote! { ::prgpu::params::convert::gpu_checkbox(filter, render_params, #enum_ident::#id) },
		Kind::Color { .. } => quote! { ::prgpu::params::convert::gpu_color(filter, render_params, #enum_ident::#id) },
		Kind::Point { .. } => quote! { ::prgpu::params::convert::gpu_point(filter, render_params, #enum_ident::#id) },
		Kind::Popup { .. } => quote! { ::prgpu::params::convert::gpu_popup(filter, render_params, #enum_ident::#id) },
		Kind::Button { .. } | Kind::Layer { .. } | Kind::Custom { .. } => return None,
	};
	Some(quote! { ::prgpu::Snapshot::set(&mut snapshot, #enum_ident::#id, #call); })
}

fn value_ty(kind: &Kind) -> TokenStream {
	match kind {
		Kind::Slider { .. } | Kind::Angle { .. } => quote! { f32 },
		Kind::Checkbox { .. } => quote! { bool },
		Kind::Color { .. } => quote! { ::prgpu::Color },
		Kind::Point { .. } => quote! { ::prgpu::Point2 },
		Kind::Popup { value_ty: PopupTy::U32, .. } => quote! { u32 },
		Kind::Popup { value_ty: PopupTy::Enum(path), .. } => quote! { #path },
		Kind::Button { .. } | Kind::Layer { .. } | Kind::Custom { .. } => quote! { () },
	}
}

fn buttons(params: &[ParamDef], enum_ident: &Ident) -> TokenStream {
	let entries: Vec<TokenStream> = params
		.iter()
		.filter_map(|p| match &p.kind {
			Kind::Button { on_click: Some(path) } => {
				let id = &p.ident;
				Some(quote! { (#enum_ident::#id, #path as fn()) })
			}
			_ => None,
		})
		.collect();
	quote! { &[ #(#entries),* ] }
}

fn debug_param(params: &[ParamDef], enum_ident: &Ident) -> TokenStream {
	for p in params {
		if p.debug_only && matches!(p.kind, Kind::Checkbox { .. }) {
			let id = &p.ident;
			return quote! { ::core::option::Option::Some(#enum_ident::#id) };
		}
	}
	quote! { ::core::option::Option::None }
}

fn flit32(v: f32) -> TokenStream {
	format!("{v}f32").parse().unwrap()
}

fn flit64(v: f32) -> TokenStream {
	format!("{v}f64").parse().unwrap()
}

fn u8lit(v: u8) -> TokenStream {
	format!("{v}u8").parse().unwrap()
}

fn i16lit(v: i16) -> TokenStream {
	format!("{v}i16").parse().unwrap()
}

fn lit_usize(v: usize) -> TokenStream {
	format!("{v}usize").parse().unwrap()
}

fn lit_u32(v: u32) -> TokenStream {
	format!("{v}u32").parse().unwrap()
}
