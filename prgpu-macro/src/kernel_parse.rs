use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{braced, Ident, Result, Token};

/// One `kernel!` invocation block: `name { field: type [= expr], ... }`.
pub struct KernelDecl {
	pub doc: Option<syn::Attribute>,
	pub name: Ident,
	pub fields: Vec<FieldDecl>,
}

pub struct FieldDecl {
	pub name: Ident,
	pub ty: syn::Type,
	pub extractor: Option<ExtractorExpr>,
}

/// Wrapper around a parsed expression kept as token stream so we can embed it
/// verbatim in codegen *and* visit it for ident collection.
pub struct ExtractorExpr {
	pub expr: syn::Expr,
}

/// Top-level input: zero or more `name { ... }` blocks.
pub struct KernelInput {
	pub decls: Vec<KernelDecl>,
}

impl Parse for KernelInput {
	fn parse(input: ParseStream<'_>) -> Result<Self> {
		let mut decls = Vec::new();
		while !input.is_empty() {
			decls.push(input.parse()?);
		}
		Ok(KernelInput { decls })
	}
}

impl Parse for KernelDecl {
	fn parse(input: ParseStream<'_>) -> Result<Self> {
		// Capture leading doc comments.
		let mut doc = None;
		for attr in input.call(syn::Attribute::parse_outer)? {
			if attr.path().is_ident("doc") {
				doc = Some(attr);
			} else {
				return Err(syn::Error::new(attr.span(), "unexpected attribute on kernel decl"));
			}
		}

		let name: Ident = input.parse()?;

		let content;
		braced!(content in input);

		let mut fields = Vec::new();
		while !content.is_empty() {
			fields.push(content.parse()?);
			// Optional trailing comma.
			if content.peek(Token![,]) {
				content.parse::<Token![,]>()?;
			}
		}

		Ok(KernelDecl { doc, name, fields })
	}
}

impl Parse for FieldDecl {
	fn parse(input: ParseStream<'_>) -> Result<Self> {
		let name: Ident = input.parse()?;
		input.parse::<Token![:]>()?;
		let ty: syn::Type = input.parse()?;

		let extractor = if input.peek(Token![=]) {
			input.parse::<Token![=]>()?;
			let expr: syn::Expr = input.parse()?;
			Some(ExtractorExpr { expr })
		} else {
			None
		};

		Ok(FieldDecl { name, ty, extractor })
	}
}

/// Collect every ident referenced in an extractor expression.
pub fn collect_idents(expr: &syn::Expr) -> Vec<Ident> {
	struct Visitor(Vec<Ident>);
	impl<'a> syn::visit::Visit<'a> for Visitor {
		fn visit_ident(&mut self, i: &'a Ident) {
			self.0.push(i.clone());
		}
	}
	let mut v = Visitor(Vec::new());
	syn::visit::visit_expr(&mut v, expr);
	v.0
}

/// Framework extractor names: these get closure bindings (`|| ctx.name()`).
pub const FRAMEWORK_EXTRACTORS: &[&str] = &["debug_view", "time_seconds", "frame_index", "progress"];

/// Rewrite a Rust type for the gpu_struct layout:
/// - `bool` → `u32`
/// - `prgpu::BlendMode` → `u32`
/// Returns `None` if no rewrite needed.
pub fn rewrite_type(ty: &syn::Type) -> Option<syn::Type> {
	if let syn::Type::Path(tp) = ty {
		let last = tp.path.segments.last().unwrap();
		if last.ident == "bool" {
			// bool → u32
			let mut u32_path: syn::Path = syn::parse_quote!(u32);
			return Some(syn::Type::Path(syn::TypePath { qself: None, path: u32_path }));
		}
		if last.ident == "BlendMode" {
			// Check for prgpu::BlendMode path prefix.
			let segs = &tp.path.segments;
			if segs.len() >= 2 {
				let prev = &segs[segs.len() - 2].ident;
				if prev == "prgpu" {
					let mut u32_path: syn::Path = syn::parse_quote!(u32);
					return Some(syn::Type::Path(syn::TypePath { qself: None, path: u32_path }));
				}
			}
		}
	}
	None
}

/// Check if a type is `prgpu::BlendMode` (needs a typed accessor).
pub fn is_blend_mode(ty: &syn::Type) -> bool {
	if let syn::Type::Path(tp) = ty {
		let segs = &tp.path.segments;
		if segs.len() >= 2 {
			let last = &segs[segs.len() - 1].ident;
			let prev = &segs[segs.len() - 2].ident;
			return last == "BlendMode" && prev == "prgpu";
		}
	}
	false
}

/// Check if a type is `bool`.
pub fn is_bool(ty: &syn::Type) -> bool {
	if let syn::Type::Path(tp) = ty {
		let last = tp.path.segments.last().unwrap();
		return last.ident == "bool";
	}
	false
}

/// Check if a type is a fixed-size array `[T; N]`.
pub fn is_array_type(ty: &syn::Type) -> bool {
	matches!(ty, syn::Type::Array(_))
}

/// Find the first marker ident (not a framework extractor) in a set of
/// extractor expressions. Returns `None` if there are no markers.
pub fn first_marker_ident(decl: &KernelDecl) -> Option<Ident> {
	for field in &decl.fields {
		if let Some(ref ext) = field.extractor {
			for ident in collect_idents(&ext.expr) {
				let name = ident.to_string();
				if !FRAMEWORK_EXTRACTORS.contains(&name.as_str()) {
					return Some(ident);
				}
			}
		}
	}
	None
}
