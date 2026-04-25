use proc_macro::TokenStream;

mod diagnostics;
mod generate;
mod layout;
mod parse;
mod types;

use types::GpuType;

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

    // Reject generic structs
    if !item_struct.generics.params.is_empty() {
        return syn::Error::new(
            item_struct.ident.span(),
            "#[gpu_struct] does not support generic structs; \
             remove generic parameters or use a concrete type",
        )
        .to_compile_error()
        .into();
    }

    // Validate existing repr attributes
    if let Err(e) = diagnostics::validate_repr(&item_struct, &config) {
        return e.to_compile_error().into();
    }

    // Extract named fields
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

    // Resolve field types
    let mut resolved_fields: Vec<(syn::Ident, GpuType, proc_macro2::Span)> = Vec::new();
    for field in fields {
        let field_name = field.ident.clone().unwrap();
        let field_span = syn::spanned::Spanned::span(&field.ty);

        // Check for #[gpu_nested] attribute on this field
        let is_gpu_nested = field.attrs.iter().any(|attr| {
            attr.path().segments.len() == 1 && attr.path().segments[0].ident == "gpu_nested"
        });

        match types::resolve_type(&field.ty, &config, is_gpu_nested) {
            Ok(gpu_type) => resolved_fields.push((field_name, gpu_type, field_span)),
            Err(e) => return e.to_compile_error().into(),
        }
    }

    // Compute layout
    let field_layouts: Vec<_> = resolved_fields
        .iter()
        .map(|(name, gpu_type, _)| (name.clone(), gpu_type.clone()))
        .collect();

    let struct_layout = layout::compute_layout(&field_layouts, config.align, &config.targets);

    // Generate output
    let output = generate::generate(&mut item_struct, &config, &struct_layout, &resolved_fields);

    output.into()
}
