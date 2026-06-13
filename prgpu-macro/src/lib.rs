use proc_macro::TokenStream;

mod diagnostics;
mod generate;
mod kernel_gen;
mod kernel_parse;
mod layout;
mod params_gen;
mod params_parse;
mod parse;
mod popup;
mod types;

use types::GpuType;

/// `params! { pub enum Params { #[slider(..)] Strength, .. } }` — see
/// `prgpu::params`. Generates the discriminant enum, per-param markers, the
/// `ParamsSpec` (registration + snapshot), and the legacy `SetupParams` bridge.
#[proc_macro]
pub fn params(item: TokenStream) -> TokenStream {
    let input = match syn::parse::<params_parse::ParamsInput>(item) {
        Ok(i) => i,
        Err(e) => return e.to_compile_error().into(),
    };
    params_gen::generate(input).into()
}

/// `kernel! { name { field: type [= expr], ... } }` — declares a kernel module
/// with GPU-laid-out params, `FromCtx` extraction, ABI check, and dispatch wiring.
#[proc_macro]
pub fn kernel(item: TokenStream) -> TokenStream {
    let input = match syn::parse::<kernel_parse::KernelInput>(item) {
        Ok(i) => i,
        Err(e) => return e.to_compile_error().into(),
    };
    kernel_gen::generate(&input.decls).into()
}

/// `#[derive(prgpu::Popup)]` on a `#[repr(u32)]` enum with `#[option("..")]`.
#[proc_macro_derive(Popup, attributes(option))]
pub fn popup(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    match popup::derive_popup(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn gpu_struct(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_tokens: proc_macro2::TokenStream = attr.into();
    let item_tokens: proc_macro2::TokenStream = item.into();

    let config = match parse::parse_config(&attr_tokens) {
        Ok(c) => c,
        Err(e) => return e.to_compile_error().into(),
    };

    let mut item_struct: syn::ItemStruct = match syn::parse2(item_tokens.clone()) {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };

    if !item_struct.generics.params.is_empty() {
        return syn::Error::new(
            item_struct.ident.span(),
            "#[gpu_struct] does not support generic structs; \
             remove generic parameters or use a concrete type",
        )
        .to_compile_error()
        .into();
    }

    if let Err(e) = diagnostics::validate_repr(&item_struct, &config) {
        return e.to_compile_error().into();
    }

    let fields = match &item_struct.fields {
        syn::Fields::Named(f) => &f.named,
        syn::Fields::Unit => {
            return syn::Error::new(
                item_struct.ident.span(),
                "#[gpu_struct] cannot be applied to unit structs; add at least one field",
            )
            .to_compile_error()
            .into();
        }
        _ => {
            return syn::Error::new(
                item_struct.ident.span(),
                "#[gpu_struct] only supports structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        if field_name.to_string().starts_with("_pad") {
            return syn::Error::new(
                field_name.span(),
                format!(
                    "manual padding field `{field_name}` — #[gpu_struct] injects padding \
                     automatically; construct with `..Default::default()` instead",
                ),
            )
            .to_compile_error()
            .into();
        }
    }

    let mut resolved_fields: Vec<(syn::Ident, GpuType, proc_macro2::Span)> = Vec::new();
    for field in fields {
        let field_name = field.ident.clone().unwrap();
        let field_span = syn::spanned::Spanned::span(&field.ty);

        let is_gpu_nested = field.attrs.iter().any(|attr| {
            attr.path().segments.len() == 1 && attr.path().segments[0].ident == "gpu_nested"
        });

        match types::resolve_type(&field.ty, &config, is_gpu_nested) {
            Ok(gpu_type) => resolved_fields.push((field_name, gpu_type, field_span)),
            Err(e) => return e.to_compile_error().into(),
        }
    }

    let field_layouts: Vec<_> = resolved_fields
        .iter()
        .map(|(name, gpu_type, _)| (name.clone(), gpu_type.clone()))
        .collect();

    let struct_layout = layout::compute_layout(&field_layouts, config.align, &config.targets);

    let output = generate::generate(&mut item_struct, &config, &struct_layout, &resolved_fields);

    output.into()
}
