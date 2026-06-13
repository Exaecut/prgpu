use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::kernel_parse::{collect_idents, first_marker_ident, is_array_type, is_blend_mode, is_bool, rewrite_type, FieldDecl, KernelDecl, FRAMEWORK_EXTRACTORS};

pub fn generate(input: &[KernelDecl]) -> TokenStream {
	let mut out = TokenStream::new();
	for decl in input {
		out.extend(generate_one(decl));
	}
	out
}

fn generate_one(decl: &KernelDecl) -> TokenStream {
	let name = &decl.name;
	let pascal = pascal_case(&name.to_string());
	let pascal_ident = Ident::new(&pascal, name.span());

	// Struct fields.
	let struct_fields = struct_fields(decl);

	// KernelParams impl.
	let kernel_params_impl = quote! {
		impl ::prgpu::KernelParams for Params {
			const SIZE: usize = <Params>::SIZE;
			const ALIGN: usize = <Params>::ALIGN;
		}
	};

	// Default impl.
	let default_impl = quote! {
		impl ::core::default::Default for Params {
			fn default() -> Self {
				<Self as ::bytemuck::Zeroable>::zeroed()
			}
		}
	};

	// ABI check.
	let abi_check = quote! {
		const _: () = {
			assert!(
				__abi::USER_PARAMS_SIZE == ::core::usize::MAX
					|| __abi::USER_PARAMS_SIZE == <Params as ::prgpu::KernelParams>::SIZE,
				"kernel params size mismatch between Rust and slangc-reflected ConstantBuffer<UserParams>"
			);
		};
	};

	// FromCtx impl.
	let from_ctx_impl = from_ctx_impl(decl);

	// SHADER const.
	let shader_const = quote! {
		#[doc(hidden)]
		pub const SHADER: &[u8] =
			::core::include_bytes!(::core::concat!(::core::env!("OUT_DIR"), "/", stringify!(#name), ".shader"));
	};

	// Popup accessors for BlendMode fields.
	let popup_accessors = popup_accessors(decl);

	// PascalCase alias at the parent module level.
	let type_alias = quote! {
		pub type #pascal_ident = #name::Params;
	};

	let name_str = name.to_string();
	let doc_attr = decl.doc.as_ref().map(|a| quote! { #a });

	quote! {
		#doc_attr
		pub mod #name {
			use super::*;

			mod __abi {
				::core::include!(::core::concat!(::core::env!("OUT_DIR"), "/", #name_str, ".abi.rs"));
			}

			#[::prgpu::gpu_struct(align = 16)]
			pub struct Params {
				#(#struct_fields,)*
			}

			#kernel_params_impl
			#default_impl
			#abi_check
			#from_ctx_impl
			#shader_const

			::prgpu::__kernel_dispatch_externs!(#name);

			#popup_accessors

			pub fn kernel() -> ::prgpu::Kernel<Params> {
				::prgpu::paste::paste! {
					::prgpu::Kernel::new(
						stringify!(#name),
						SHADER,
						stringify!(#name),
						[<#name _cpu_dispatch>],
						[<#name _cpu_dispatch_tile>],
					)
				}
			}
		}

		#type_alias
	}
}

fn struct_fields(decl: &KernelDecl) -> Vec<TokenStream> {
	decl.fields.iter().map(|f| struct_field(f)).collect()
}

fn struct_field(f: &FieldDecl) -> TokenStream {
	let name = &f.name;
	let ty = if let Some(rewritten) = rewrite_type(&f.ty) {
		quote! { #rewritten }
	} else {
		let ty = &f.ty;
		quote! { #ty }
	};
	quote! {
		pub #name: #ty
	}
}

fn from_ctx_impl(decl: &KernelDecl) -> TokenStream {
	let marker = first_marker_ident(decl);

	// Build mapping: original ident → prefixed ident.
	let mut seen = std::collections::HashSet::new();
	let mut ident_map: Vec<(Ident, Ident)> = Vec::new();
	for field in &decl.fields {
		if let Some(ref ext) = field.extractor {
			for ident in collect_idents(&ext.expr) {
				let key = ident.to_string();
				if !seen.insert(key.clone()) {
					continue;
				}
				if FRAMEWORK_EXTRACTORS.contains(&key.as_str()) {
					ident_map.push((ident.clone(), ident.clone()));
				} else if is_likely_marker(&key) {
					let prefixed = Ident::new(&format!("__prgpu_{key}"), ident.span());
					ident_map.push((ident, prefixed));
				}
				// Else: skip sugar — resolve from module scope.
			}
		}
	}

	// Generate prelude bindings.
	let mut prelude_bindings = Vec::new();
	for (orig, prefixed) in &ident_map {
		let key = orig.to_string();
		if FRAMEWORK_EXTRACTORS.contains(&key.as_str()) {
			prelude_bindings.push(quote! {
				let #prefixed = || ctx.#prefixed();
			});
		} else {
			prelude_bindings.push(quote! {
				let #prefixed = ctx.get(#orig);
			});
		}
	}

	// Build the struct literal fields with rewritten expressions.
	let mut field_inits = Vec::new();
	for field in &decl.fields {
		let name = &field.name;
		let init = match &field.extractor {
			Some(ext) => {
				let rewritten = rewrite_expr_idents(&ext.expr, &ident_map);
				if is_bool(&field.ty) || is_blend_mode(&field.ty) {
					quote! { #rewritten as u32 }
				} else if is_array_type(&field.ty) {
					quote! { ::core::convert::Into::into(#rewritten) }
				} else {
					quote! { #rewritten }
				}
			}
			None => {
				quote! { ::core::default::Default::default() }
			}
		};
		field_inits.push(quote! {
			#name: #init
		});
	}

	// Spec type: use the first marker if present.
	let spec_ty = if let Some(ref m) = marker {
		quote! { <#m as ::prgpu::Param>::Spec }
	} else {
		return from_ctx_impl_no_markers(decl, &prelude_bindings, &field_inits);
	};

	let body = quote! {
		#(#prelude_bindings)*
		Self {
			#(#field_inits,)*
			..::core::default::Default::default()
		}
	};

	quote! {
		impl ::prgpu::FromCtx for Params {
			type Spec = #spec_ty;

			fn from_ctx(ctx: &::prgpu::Ctx<Self::Spec>) -> Self {
				#body
			}
		}
	}
}

fn from_ctx_impl_no_markers(decl: &KernelDecl, prelude_bindings: &[TokenStream], field_inits: &[TokenStream]) -> TokenStream {
	let body = quote! {
		#(#prelude_bindings)*
		Self {
			#(#field_inits,)*
			..::core::default::Default::default()
		}
	};

	quote! {
		impl Params {
			#[doc(hidden)]
			pub fn from_ctx<S: ::prgpu::ParamsSpec>(ctx: &::prgpu::Ctx<S>) -> Self {
				#body
			}
		}
	}
}

/// Replace idents in an expression using the given mapping.
fn rewrite_expr_idents(expr: &syn::Expr, map: &[(Ident, Ident)]) -> TokenStream {
	// Build a lookup from string to replacement.
	let lookup: std::collections::HashMap<String, &Ident> = map
		.iter()
		.map(|(orig, prefixed)| (orig.to_string(), prefixed))
		.collect();

	struct ReplaceVisitor<'a> {
		lookup: &'a std::collections::HashMap<String, &'a Ident>,
	}

	impl<'a> syn::visit_mut::VisitMut for ReplaceVisitor<'a> {
		fn visit_ident_mut(&mut self, i: &mut Ident) {
			if let Some(replacement) = self.lookup.get(&i.to_string()) {
				*i = (*replacement).clone();
			}
		}
	}

	let mut expr = expr.clone();
	let mut v = ReplaceVisitor { lookup: &lookup };
	syn::visit_mut::visit_expr_mut(&mut v, &mut expr);
	quote! { #expr }
}

fn popup_accessors(decl: &KernelDecl) -> TokenStream {
	let mut accessors = TokenStream::new();
	for field in &decl.fields {
		if is_blend_mode(&field.ty) {
			let name = &field.name;
			accessors.extend(quote! {
				impl Params {
					#[inline]
					pub fn #name(&self) -> ::prgpu::BlendMode {
						<::prgpu::BlendMode as ::prgpu::PopupOptions>::from_index(self.#name)
					}
				}
			});
		}
	}
	accessors
}

fn pascal_case(s: &str) -> String {
	let mut result = String::with_capacity(s.len());
	let mut capitalize = true;
	for ch in s.chars() {
		if ch == '_' {
			capitalize = true;
		} else if capitalize {
			result.push(ch.to_ascii_uppercase());
			capitalize = false;
		} else {
			result.push(ch);
		}
	}
	result
}

/// Only PascalCase idents (no underscores, starts uppercase) get marker sugar.
/// SCREAMING_CASE (DEG_TO_RAD) and snake_case resolve from module scope.
fn is_likely_marker(name: &str) -> bool {
	name.starts_with(|c: char| c.is_ascii_uppercase()) && !name.contains('_')
}
