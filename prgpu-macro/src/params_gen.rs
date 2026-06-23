//! Codegen for `params!`: the discriminant enum, one marker per param, the
//! `ParamsSpec` (host registration + per-frame `Snapshot`), the `Snapshot`
//! storage, the auto-generated `Router` enum + per-instance route store, and
//! the transitional `SetupParams` bridge.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::params_parse::{Kind, LabelExpr, Node, ParamDef, ParamsInput, PopupSource, PopupTy, marker_end, marker_start};

pub fn generate(input: ParamsInput) -> TokenStream {
	let ParamsInput {
		vis,
		enum_ident,
		mut variants,
		mut params,
		nodes,
		routes,
		initial_route,
	} = input;

	// Auto-inject a hidden, per-instance popup that stores the active route.
	// Added last so existing params keep their discriminants/indices.
	let has_routes = !routes.is_empty();
	let route_ident = Ident::new("__Route", Span::call_site());
	let initial_index = initial_route
		.as_ref()
		.and_then(|r| routes.iter().position(|x| x == r))
		.unwrap_or(0) as u32;
	if has_routes {
		let route_names: Vec<String> = routes.iter().map(|r| r.to_string()).collect();
		variants.push(route_ident.clone());
		params.push(ParamDef {
			ident: route_ident.clone(),
			label: LabelExpr::Str(String::new()),
			kind: Kind::RouteStore { routes: route_names, initial: initial_index },
			debug_only: false,
		});
	}

	let n = variants.len();
	let count = lit_usize(n);

	// `ALL` is for iterate-and-toggle visibility; only real, non-route leaf
	// params, never group markers or the hidden route store.
	let all_param_idents: Vec<&Ident> = params
		.iter()
		.filter(|p| !matches!(p.kind, Kind::RouteStore { .. }))
		.map(|p| &p.ident)
		.collect();

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

	let label_params: Vec<&Ident> = params.iter().filter(|p| matches!(p.kind, Kind::Label { .. })).map(|p| &p.ident).collect();
	let label_params_tokens = quote! { &[ #( #enum_ident::#label_params ),* ] };

	// `#[button(text = expr)]` or `#[button(label = expr)]` → buttons whose
	// caption is rewritten live via PF_UpdateParamUI (the Warp Stabilizer
	// pattern), as opposed to custom-draw `#[label]` text. A dynamic `label`
	// is treated the same way as `text` — the expression is re-evaluated each
	// refresh and pushed through `set_name`.
	let name_driven_params: Vec<&Ident> = params
		.iter()
		.filter(|p| match (&p.kind, &p.label) {
			(Kind::Button { text: Some(_), .. }, _) => true,
			(Kind::Button { .. }, LabelExpr::Expr(_)) => true,
			_ => false,
		})
		.map(|p| &p.ident)
		.collect();
	let name_driven_params_tokens = quote! { &[ #( #enum_ident::#name_driven_params ),* ] };

	// `#[label(text = expr)]`, `#[button(text = expr)]`, and
	// `#[button(label = expr)]` → declarative text bindings. All flow through
	// the same `label_rules`; the adapter routes name-driven buttons to
	// `set_name`+`update_param_ui` and labels to the draw stash. `text` wins
	// over a dynamic `label` when both are present.
	let label_text_bindings: Vec<TokenStream> = params
		.iter()
		.filter_map(|p| {
			let expr = match (&p.kind, &p.label) {
				(Kind::Label { text: Some(expr) }, _) => expr,
				(Kind::Button { text: Some(expr), .. }, _) => expr,
				(Kind::Button { .. }, LabelExpr::Expr(expr)) => expr,
				_ => return None,
			};
			let id = &p.ident;
			Some(quote! {
				ui.set_label_id(
					#enum_ident::#id,
					|_ctx| ::core::convert::Into::<::std::string::String>::into(#expr),
				);
			})
		})
		.collect();

	// `#[button(disabled = expr)]` → the adapter toggles `PF_PUI_DISABLED`
	// each UI tick when the expr is `true`; `disabled = true` is the static form.
	let disabled_bindings: Vec<TokenStream> = params
		.iter()
		.filter_map(|p| {
			let expr = match &p.kind {
				Kind::Button { disabled: Some(expr), .. } => expr,
				_ => return None,
			};
			let id = &p.ident;
			Some(quote! {
				ui.set_disabled_id(#enum_ident::#id, |_ctx| #expr);
			})
		})
		.collect();

	// `(group-start marker, route index)` for every routed group.
	let mut routed_groups: Vec<TokenStream> = Vec::new();
	collect_routed_groups(&nodes, &routes, &enum_ident, &mut routed_groups);
	let route_param_tokens = if has_routes {
		quote! { ::core::option::Option::Some(#enum_ident::#route_ident) }
	} else {
		quote! { ::core::option::Option::None }
	};

	let router_def = router_def(&vis, &routes, &initial_route);

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
	// The injected route store isn't in the group tree; register it explicitly.
	let route_reg = if has_routes {
		let route_param = params.iter().find(|p| matches!(p.kind, Kind::RouteStore { .. })).unwrap();
		reg_stmt(route_param, &enum_ident)
	} else {
		quote! {}
	};
	let cpu_stmts: Vec<TokenStream> = params.iter().filter_map(|p| cpu_stmt(p, &enum_ident)).collect();
	let gpu_stmts: Vec<TokenStream> = params.iter().filter_map(|p| gpu_stmt(p, &enum_ident)).collect();

	let buttons = buttons(&params, &enum_ident);
	let debug_param = debug_param(&params, &enum_ident);

	quote! {
		#enum_def

		#( #markers )*

		#snapshot

		#router_def

		impl ::prgpu::ParamsSpec for #enum_ident {
			const COUNT: usize = #count;
			const ALL: &'static [Self] = &[ #( #enum_ident::#all_param_idents ),* ];
			const DEBUG_PARAM: ::core::option::Option<Self> = #debug_param;
			const LAYER_PARAMS: &'static [Self] = #layer_params_tokens;
			const LABEL_PARAMS: &'static [Self] = #label_params_tokens;
			const NAME_DRIVEN_PARAMS: &'static [Self] = #name_driven_params_tokens;
			const ROUTED_GROUPS: &'static [(Self, u32)] = &[ #( #routed_groups ),* ];
			const ROUTE_PARAM: ::core::option::Option<Self> = #route_param_tokens;
			type Snapshot = __Snapshot;

			#[allow(unused_variables)]
			fn register(params: &mut ::after_effects::Parameters<Self>) -> ::core::result::Result<(), ::after_effects::Error> {
				#route_reg
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

			fn buttons() -> &'static [(Self, fn(&mut ::prgpu::ActionCtx<Self>))] {
				#buttons
			}

			#[allow(unused_variables)]
			fn contribute_labels(ui: &mut ::prgpu::Ui<Self>) {
				#( #label_text_bindings )*
				#( #disabled_bindings )*
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

/// Emit the `pub enum Router` + `impl Router` + `impl prgpu::Route`. Empty when
/// no group declares a `route`.
fn router_def(vis: &syn::Visibility, routes: &[Ident], initial: &Option<Ident>) -> TokenStream {
	if routes.is_empty() {
		return quote! {};
	}
	let names: Vec<String> = routes.iter().map(|r| r.to_string()).collect();
	let indices: Vec<u32> = (0..routes.len() as u32).collect();
	let initial_variant = initial.clone().unwrap_or_else(|| routes[0].clone());
	let count = routes.len();
	let count_lit = lit_usize(count);
	quote! {
		#[repr(u32)]
		#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
		#vis enum Router {
			#( #routes = #indices, )*
		}

		impl Router {
			/// Number of routes.
			pub const COUNT: usize = #count_lit;

			/// Active route for the instance being processed.
			pub fn current() -> Router {
				<Router as ::prgpu::Route>::from_index(::prgpu::effect::route::current_index())
			}
			/// Request a route change. Applied after the current handler returns.
			pub fn set(route: Router) {
				::prgpu::effect::route::request_index(<Router as ::prgpu::Route>::to_index(route));
			}
			/// The route's variant name.
			pub fn name(self) -> &'static str {
				<Router as ::prgpu::Route>::name(self)
			}
			/// Zero-based position of this route.
			pub fn index(self) -> u32 {
				self as u32
			}
			/// Next route, wrapping.
			pub fn next(self) -> Router {
				<Router as ::prgpu::Route>::from_index((self as u32 + 1) % (#count_lit as u32))
			}
			/// Previous route, wrapping.
			pub fn prev(self) -> Router {
				let n = #count_lit as u32;
				<Router as ::prgpu::Route>::from_index((self as u32 + n - 1) % n)
			}
		}

		impl ::prgpu::Route for Router {
			const ALL: &'static [Router] = &[ #( Router::#routes ),* ];
			const INITIAL: Router = Router::#initial_variant;
			fn to_index(self) -> u32 { self as u32 }
			fn from_index(index: u32) -> Self {
				match index {
					#( #indices => Router::#routes, )*
					_ => Router::#initial_variant,
				}
			}
			fn name(self) -> &'static str {
				match self {
					#( Router::#routes => #names, )*
				}
			}
		}
	}
}

fn collect_routed_groups(nodes: &[Node], routes: &[Ident], enum_ident: &Ident, out: &mut Vec<TokenStream>) {
	for node in nodes {
		if let Node::Group { idx, route, children, .. } = node {
			if let Some(r) = route {
				if let Some(ri) = routes.iter().position(|x| x == r) {
					let marker = marker_start(*idx);
					let ri = lit_u32(ri as u32);
					out.push(quote! { (#enum_ident::#marker, #ri) });
				}
			}
			collect_routed_groups(children, routes, enum_ident, out);
		}
	}
}

fn emit_node(node: &Node, enum_ident: &Ident, params: &[ParamDef]) -> TokenStream {
	match node {
		Node::Param(i) => reg_stmt(&params[*i], enum_ident),
		Node::Group { idx, name, collapsed, children, .. } => {
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
	let label = label_str(&p.label);
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
			let flags = quote! {
				p.set_flag(::after_effects::ParamFlag::SUPERVISE, true);
				p.set_flag(::after_effects::ParamFlag::START_COLLAPSED, true);
				-1
			};
			match &p.label {
				LabelExpr::Str(s) => quote! {
					params.add_customized(#enum_ident::#id, #s, ::after_effects::ButtonDef::setup(|f| {
						f.set_label(#s);
					}), |p| { #flags })?;
				},
				LabelExpr::Expr(e) => quote! {
					let __label: ::std::string::String =
						::core::convert::Into::<::std::string::String>::into(#e);
					params.add_customized(#enum_ident::#id, __label.as_str(), ::after_effects::ButtonDef::setup(|f| {
						f.set_label(__label.as_str());
					}), |p| { #flags })?;
				},
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
		Kind::Label { .. } => {
			quote! {
				// Premiere only sends PF_Event_DRAW to custom-UI params that are
				// ARBITRARY (or null) — and empirically only the arbitrary form
				// actually fires (Custom_ECW_UI.cpp uses PF_ADD_ARBITRARY2 on its
				// Premiere branch). The payload is a throwaway LabelArb; the text
				// is drawn from the label stash. CONTROL + ui dims give it a
				// drawable control area.
				params.add_customized(#enum_ident::#id, "",
					::after_effects::ArbitraryDef::setup(|f| {
						let _ = f.set_default::<::prgpu::LabelArb>(::core::default::Default::default());
					}),
					|p| {
						p.set_flag(::after_effects::ParamFlag::SUPERVISE, true);
						p.set_ui_flag(::after_effects::ParamUIFlags::CONTROL, true);
						// Don't let the host erase (black-fill) the control area;
						// we paint only the text, leaving the panel background.
						p.set_ui_flag(::after_effects::ParamUIFlags::DO_NOT_ERASE_CONTROL, true);
						p.set_ui_width(200);
						p.set_ui_height(20);
						-1
					})?;
			}
		}
		Kind::RouteStore { routes, initial } => {
			let opts: Vec<&String> = routes.iter().collect();
			let default1 = lit_i32(*initial as i32 + 1);
			quote! {
				params.add_customized(#enum_ident::#id, "", ::after_effects::PopupDef::setup(|f| {
					f.set_options(&[ #(#opts),* ]);
					f.set_default(#default1);
					f.set_value(#default1);
				}), |p| {
					p.set_ui_flag(::after_effects::ParamUIFlags::INVISIBLE, true);
					-1
				})?;
			}
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
		Kind::Button { .. } | Kind::Layer { .. } | Kind::Custom { .. } | Kind::Label { .. } | Kind::RouteStore { .. } => return None,
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
		Kind::Button { .. } | Kind::Layer { .. } | Kind::Custom { .. } | Kind::Label { .. } | Kind::RouteStore { .. } => return None,
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
		Kind::Button { .. } | Kind::Layer { .. } | Kind::Custom { .. } | Kind::Label { .. } | Kind::RouteStore { .. } => quote! { () },
	}
}

fn buttons(params: &[ParamDef], enum_ident: &Ident) -> TokenStream {
	let entries: Vec<TokenStream> = params
		.iter()
		.filter_map(|p| {
			let id = &p.ident;
			match &p.kind {
				// Context-aware handler: passed through directly.
				Kind::Button { on_action: Some(path), .. } => Some(quote! {
					(#enum_ident::#id, #path as fn(&mut ::prgpu::ActionCtx<#enum_ident>))
				}),
				// Legacy fn() handler: wrapped to ignore the context.
				Kind::Button { on_click: Some(path), on_action: None, .. } => Some(quote! {
					(#enum_ident::#id, {
						fn __wrap(_cx: &mut ::prgpu::ActionCtx<#enum_ident>) { #path(); }
						__wrap as fn(&mut ::prgpu::ActionCtx<#enum_ident>)
					})
				}),
				_ => None,
			}
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

fn lit_i32(v: i32) -> TokenStream {
	format!("{v}i32").parse().unwrap()
}

fn lit_usize(v: usize) -> TokenStream {
	format!("{v}usize").parse().unwrap()
}

fn lit_u32(v: u32) -> TokenStream {
	format!("{v}u32").parse().unwrap()
}

fn label_str(label: &LabelExpr) -> String {
	match label {
		LabelExpr::Str(s) => s.clone(),
		LabelExpr::Expr(_) => String::new(),
	}
}
