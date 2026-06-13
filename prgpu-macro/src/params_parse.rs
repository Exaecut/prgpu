//! Parser for `params! { pub enum Params { #[kind(..)] Variant, .. } }`.
//!
//! Produces the flat registration order (`variants`, including synthesized
//! group-start/end markers), the per-param definitions, and a group tree the
//! codegen walks to emit `add_group` closures.

use proc_macro2::Span;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Error, Expr, Ident, LitBool, LitStr, Path, Token, Visibility, braced};

pub struct ParamsInput {
	pub vis: Visibility,
	pub enum_ident: Ident,
	pub variants: Vec<Ident>,
	pub params: Vec<ParamDef>,
	pub nodes: Vec<Node>,
}

pub enum Node {
	Param(usize),
	Group { idx: usize, name: String, collapsed: bool, children: Vec<Node> },
}

pub struct ParamDef {
	pub ident: Ident,
	pub label: String,
	pub kind: Kind,
	pub debug_only: bool,
}

pub enum PopupTy {
	U32,
	Enum(Path),
}

pub enum PopupSource {
	Inline(Vec<String>),
	Enum(Path),
}

pub enum Kind {
	Slider {
		vmin: f32,
		vmax: f32,
		default: f32,
		smin: f32,
		smax: f32,
		percent: bool,
		precision: Option<i16>,
	},
	Checkbox {
		default: bool,
		supervise: bool,
	},
	Color {
		r: u8,
		g: u8,
		b: u8,
		a: u8,
	},
	Angle {
		default: f32,
	},
	Point {
		x: f32,
		y: f32,
	},
	Popup {
		options: PopupSource,
		default: Expr,
		value_ty: PopupTy,
	},
	Button {
		on_click: Option<Path>,
	},
	Custom {
		setup: Path,
	},
}

pub fn marker_start(idx: usize) -> Ident {
	Ident::new(&format!("__Group{idx}Start"), Span::call_site())
}

pub fn marker_end(idx: usize) -> Ident {
	Ident::new(&format!("__Group{idx}End"), Span::call_site())
}

struct GroupFrame {
	idx: usize,
	name: String,
	collapsed: bool,
	children: Vec<Node>,
}

fn push_node(stack: &mut [GroupFrame], root: &mut Vec<Node>, node: Node) {
	match stack.last_mut() {
		Some(frame) => frame.children.push(node),
		None => root.push(node),
	}
}

enum GroupAttr {
	Open { name: String, collapsed: bool },
	End,
}

impl Parse for ParamsInput {
	fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
		let vis: Visibility = input.parse()?;
		input.parse::<Token![enum]>()?;
		let enum_ident: Ident = input.parse()?;

		let content;
		braced!(content in input);

		let mut params: Vec<ParamDef> = Vec::new();
		let mut variants: Vec<Ident> = Vec::new();
		let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
		let mut root: Vec<Node> = Vec::new();
		let mut stack: Vec<GroupFrame> = Vec::new();
		let mut group_count = 0usize;

		while !content.is_empty() {
			let attrs = content.call(Attribute::parse_outer)?;

			let mut kind_attr: Option<Attribute> = None;
			for attr in attrs {
				let id = attr.path().get_ident().map(|i| i.to_string()).unwrap_or_default();
				match id.as_str() {
					"group" => match parse_group_attr(&attr)? {
						GroupAttr::End => {
							let frame = stack
								.pop()
								.ok_or_else(|| Error::new_spanned(&attr, "`#[group(end)]` without a matching `#[group(\"...\")]`"))?;
							variants.push(marker_end(frame.idx));
							let node = Node::Group {
								idx: frame.idx,
								name: frame.name,
								collapsed: frame.collapsed,
								children: frame.children,
							};
							push_node(&mut stack, &mut root, node);
						}
						GroupAttr::Open { name, collapsed } => {
							let idx = group_count;
							group_count += 1;
							variants.push(marker_start(idx));
							stack.push(GroupFrame {
								idx,
								name,
								collapsed,
								children: Vec::new(),
							});
						}
					},
					"slider" | "checkbox" | "color" | "angle" | "point" | "popup" | "blend_mode" | "button" | "custom" => {
						if kind_attr.is_some() {
							return Err(Error::new_spanned(&attr, "more than one parameter-kind attribute on a variant"));
						}
						kind_attr = Some(attr);
					}
					other => {
						return Err(Error::new_spanned(
							&attr,
							format!("unknown attribute `{other}`; expected one of slider/checkbox/color/angle/point/popup/blend_mode/button/custom/group"),
						));
					}
				}
			}

			if content.is_empty() {
				if kind_attr.is_some() {
					return Err(Error::new(Span::call_site(), "trailing parameter-kind attribute with no variant name"));
				}
				break;
			}

			let ident: Ident = content.parse()?;
			if content.peek(Token![,]) {
				content.parse::<Token![,]>()?;
			}

			let kind_attr =
				kind_attr.ok_or_else(|| Error::new_spanned(&ident, format!("`{ident}` is missing a parameter-kind attribute (e.g. `#[slider(..)]`)")))?;
			let (kind, label, debug_only) = parse_kind(&kind_attr, &ident)?;

			if !seen.insert(ident.to_string()) {
				return Err(Error::new_spanned(&ident, format!("duplicate parameter `{ident}`")));
			}

			let param_index = params.len();
			params.push(ParamDef { ident: ident.clone(), label, kind, debug_only });
			variants.push(ident);
			push_node(&mut stack, &mut root, Node::Param(param_index));
		}

		if let Some(frame) = stack.last() {
			return Err(Error::new(Span::call_site(), format!("unclosed `#[group(\"{}\")]` — add `#[group(end)]`", frame.name)));
		}

		if variants.is_empty() {
			return Err(Error::new(Span::call_site(), "`params!` needs at least one parameter"));
		}

		Ok(ParamsInput {
			vis,
			enum_ident,
			variants,
			params,
			nodes: root,
		})
	}
}

fn parse_group_attr(attr: &Attribute) -> syn::Result<GroupAttr> {
	attr.parse_args_with(|input: ParseStream<'_>| {
		if input.peek(LitStr) {
			let name: LitStr = input.parse()?;
			let mut collapsed = true;
			if input.peek(Token![,]) {
				input.parse::<Token![,]>()?;
				let key: Ident = input.parse()?;
				if key != "collapsed" {
					return Err(Error::new_spanned(&key, "expected `collapsed`"));
				}
				if input.peek(Token![=]) {
					input.parse::<Token![=]>()?;
					let b: LitBool = input.parse()?;
					collapsed = b.value;
				}
			}
			Ok(GroupAttr::Open { name: name.value(), collapsed })
		} else {
			let id: Ident = input.parse()?;
			if id == "end" {
				Ok(GroupAttr::End)
			} else {
				Err(Error::new_spanned(&id, "expected `\"Group Name\"` or `end`"))
			}
		}
	})
}

struct KeyVals(Vec<(Ident, Option<Expr>)>);

impl Parse for KeyVals {
	fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
		let mut v = Vec::new();
		while !input.is_empty() {
			let key: Ident = input.parse()?;
			let val = if input.peek(Token![=]) {
				input.parse::<Token![=]>()?;
				Some(input.parse::<Expr>()?)
			} else {
				None
			};
			v.push((key, val));
			if input.peek(Token![,]) {
				input.parse::<Token![,]>()?;
			}
		}
		Ok(KeyVals(v))
	}
}

impl KeyVals {
	fn get(&self, key: &str) -> Option<&Expr> {
		self.0.iter().find(|(k, _)| k == key).and_then(|(_, e)| e.as_ref())
	}

	fn flag(&self, key: &str) -> bool {
		self.0.iter().any(|(k, e)| k == key && e.is_none())
	}
}

fn parse_kind(attr: &Attribute, ident: &Ident) -> syn::Result<(Kind, String, bool)> {
	let kindname = attr.path().get_ident().unwrap().to_string();
	let kv: KeyVals = attr.parse_args().unwrap_or(KeyVals(Vec::new()));
	let debug_only = kv.flag("debug_only");

	let require_label = || -> syn::Result<String> {
		kv.get("label")
			.ok_or_else(|| Error::new_spanned(ident, format!("`{ident}` is missing `label = \"...\"`")))
			.and_then(lit_str)
	};

	let kind = match kindname.as_str() {
		"slider" => {
			let (vmin, vmax) = range_f32(kv.get("range").ok_or_else(|| Error::new_spanned(ident, "slider needs `range = a..=b`"))?)?;
			let default = lit_f32(kv.get("default").ok_or_else(|| Error::new_spanned(ident, "slider needs `default = ..`"))?)?;
			if default < vmin || default > vmax {
				return Err(Error::new_spanned(ident, format!("default {default} is outside range {vmin}..={vmax}")));
			}
			let (smin, smax) = match kv.get("slider_range") {
				Some(e) => range_f32(e)?,
				None => (vmin, vmax),
			};
			let precision = match kv.get("precision") {
				Some(e) => Some(lit_i16(e)?),
				None => None,
			};
			Kind::Slider {
				vmin,
				vmax,
				default,
				smin,
				smax,
				percent: kv.flag("percent"),
				precision,
			}
		}
		"checkbox" => Kind::Checkbox {
			default: match kv.get("default") {
				Some(e) => lit_bool(e)?,
				None => false,
			},
			supervise: kv.flag("supervise"),
		},
		"color" => {
			let (r, g, b, a) = parse_hex(kv.get("default").ok_or_else(|| Error::new_spanned(ident, "color needs `default = \"#RRGGBB\"`"))?)?;
			Kind::Color { r, g, b, a }
		}
		"angle" => Kind::Angle {
			default: match kv.get("default") {
				Some(e) => lit_f32(e)?,
				None => 0.0,
			},
		},
		"point" => {
			let (x, y) = tuple_f32x2(kv.get("default").ok_or_else(|| Error::new_spanned(ident, "point needs `default = (x, y)`"))?)?;
			Kind::Point { x, y }
		}
		"popup" => {
			let options_expr = kv.get("options").ok_or_else(|| Error::new_spanned(ident, "popup needs `options = [..]` or an enum path"))?;
			let default = kv.get("default").cloned().ok_or_else(|| Error::new_spanned(ident, "popup needs `default = ..`"))?;
			let (options, value_ty) = match options_expr {
				Expr::Array(_) => {
					let labels = str_array(options_expr)?;
					(PopupSource::Inline(labels), PopupTy::U32)
				}
				Expr::Path(p) => (PopupSource::Enum(p.path.clone()), PopupTy::Enum(p.path.clone())),
				_ => return Err(Error::new_spanned(options_expr, "popup `options` must be a string array or an enum path")),
			};
			Kind::Popup { options, default, value_ty }
		}
		"blend_mode" => {
			let variant: Path = path_expr(kv.get("default").ok_or_else(|| Error::new_spanned(ident, "blend_mode needs `default = Variant`"))?)?;
			let variant_ident = variant.segments.last().map(|s| s.ident.clone()).ok_or_else(|| Error::new_spanned(ident, "invalid blend_mode default"))?;
			let blend: Path = syn::parse_quote!(::prgpu::BlendMode);
			let default: Expr = syn::parse_quote!(::prgpu::BlendMode::#variant_ident);
			Kind::Popup {
				options: PopupSource::Enum(blend.clone()),
				default,
				value_ty: PopupTy::Enum(blend),
			}
		}
		"button" => Kind::Button {
			on_click: match kv.get("on_click") {
				Some(e) => Some(path_expr(e)?),
				None => None,
			},
		},
		"custom" => Kind::Custom {
			setup: path_expr(kv.get("setup").ok_or_else(|| Error::new_spanned(ident, "custom needs `setup = path`"))?)?,
		},
		_ => unreachable!(),
	};

	let label = if matches!(kind, Kind::Custom { .. }) {
		String::new()
	} else {
		require_label()?
	};

	Ok((kind, label, debug_only))
}

fn lit_str(e: &Expr) -> syn::Result<String> {
	if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = e {
		Ok(s.value())
	} else {
		Err(Error::new_spanned(e, "expected a string literal"))
	}
}

fn lit_bool(e: &Expr) -> syn::Result<bool> {
	if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Bool(b), .. }) = e {
		Ok(b.value)
	} else {
		Err(Error::new_spanned(e, "expected `true` or `false`"))
	}
}

fn lit_f32(e: &Expr) -> syn::Result<f32> {
	match e {
		Expr::Lit(syn::ExprLit { lit: syn::Lit::Float(f), .. }) => f.base10_parse::<f32>(),
		Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) => Ok(i.base10_parse::<i64>()? as f32),
		Expr::Unary(syn::ExprUnary { op: syn::UnOp::Neg(_), expr, .. }) => Ok(-lit_f32(expr)?),
		Expr::Group(g) => lit_f32(&g.expr),
		Expr::Paren(p) => lit_f32(&p.expr),
		_ => Err(Error::new_spanned(e, "expected a numeric literal")),
	}
}

fn lit_i16(e: &Expr) -> syn::Result<i16> {
	if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. }) = e {
		i.base10_parse::<i16>()
	} else {
		Err(Error::new_spanned(e, "expected an integer literal"))
	}
}

fn range_f32(e: &Expr) -> syn::Result<(f32, f32)> {
	if let Expr::Range(r) = e {
		let start = r.start.as_ref().ok_or_else(|| Error::new_spanned(e, "range needs a start"))?;
		let end = r.end.as_ref().ok_or_else(|| Error::new_spanned(e, "range needs an end"))?;
		Ok((lit_f32(start)?, lit_f32(end)?))
	} else {
		Err(Error::new_spanned(e, "expected `a..=b`"))
	}
}

fn tuple_f32x2(e: &Expr) -> syn::Result<(f32, f32)> {
	if let Expr::Tuple(t) = e {
		if t.elems.len() == 2 {
			return Ok((lit_f32(&t.elems[0])?, lit_f32(&t.elems[1])?));
		}
	}
	Err(Error::new_spanned(e, "expected `(x, y)`"))
}

fn str_array(e: &Expr) -> syn::Result<Vec<String>> {
	if let Expr::Array(a) = e {
		a.elems.iter().map(lit_str).collect()
	} else {
		Err(Error::new_spanned(e, "expected a string array"))
	}
}

fn path_expr(e: &Expr) -> syn::Result<Path> {
	if let Expr::Path(p) = e {
		Ok(p.path.clone())
	} else {
		Err(Error::new_spanned(e, "expected a path"))
	}
}

/// `#RRGGBB` or `#RRGGBBAA` → 8-bit channels, alpha defaults to 255.
fn parse_hex(e: &Expr) -> syn::Result<(u8, u8, u8, u8)> {
	let s = lit_str(e)?;
	let hex = s.strip_prefix('#').unwrap_or(&s);
	let byte = |i: usize| -> syn::Result<u8> {
		u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| Error::new_spanned(e, "invalid hex colour"))
	};
	match hex.len() {
		6 => Ok((byte(0)?, byte(2)?, byte(4)?, 255)),
		8 => Ok((byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
		_ => Err(Error::new_spanned(e, "colour must be `#RRGGBB` or `#RRGGBBAA`")),
	}
}
