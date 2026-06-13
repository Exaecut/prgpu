use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr};

/// `#[derive(prgpu::Popup)]` on a `#[repr(u32)]` enum with `#[option("Label")]`
/// per variant. Generates `PopupOptions` (LABELS + index round-trip, clamping
/// out-of-range to the first variant) and `FromParamValue`.
pub fn derive_popup(input: &DeriveInput) -> Result<TokenStream, syn::Error> {
	let name = &input.ident;

	let Data::Enum(data) = &input.data else {
		return Err(syn::Error::new_spanned(name, "#[derive(Popup)] only supports enums"));
	};

	let mut labels = Vec::new();
	let mut variants = Vec::new();
	for v in &data.variants {
		if !matches!(v.fields, Fields::Unit) {
			return Err(syn::Error::new_spanned(v, "#[derive(Popup)] requires unit variants"));
		}
		let mut label = None;
		for attr in &v.attrs {
			if attr.path().is_ident("option") {
				let lit: LitStr = attr.parse_args()?;
				label = Some(lit.value());
			}
		}
		labels.push(label.unwrap_or_else(|| v.ident.to_string()));
		variants.push(v.ident.clone());
	}

	if variants.is_empty() {
		return Err(syn::Error::new_spanned(name, "#[derive(Popup)] enum needs at least one variant"));
	}

	let first = &variants[0];
	let arms = variants.iter().map(|v| quote! { x if x == (#name::#v as u32) => #name::#v, });

	Ok(quote! {
		impl ::prgpu::PopupOptions for #name {
			const LABELS: &'static [&'static str] = &[ #(#labels),* ];

			fn from_index(index: u32) -> Self {
				match index {
					#(#arms)*
					_ => #name::#first,
				}
			}

			fn to_index(self) -> u32 {
				self as u32
			}
		}

		impl ::prgpu::FromParamValue for #name {
			fn from_value(value: ::prgpu::ParamValue) -> Self {
				match value {
					::prgpu::ParamValue::Index(i) => <#name as ::prgpu::PopupOptions>::from_index(i),
					// Premiere GPU popups arrive as Float32.
					::prgpu::ParamValue::Float(f) => <#name as ::prgpu::PopupOptions>::from_index(f.max(0.0) as u32),
					_ => <#name as ::prgpu::PopupOptions>::from_index(0),
				}
			}
		}
	})
}
