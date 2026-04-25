use crate::layout::StructLayout;
use crate::parse::GpuStructConfig;
use crate::types::GpuType;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Ident, ItemStruct};

pub fn generate(
    item_struct: &mut ItemStruct,
    config: &GpuStructConfig,
    layout: &StructLayout,
    resolved_fields: &[(Ident, GpuType, proc_macro2::Span)],
) -> TokenStream2 {
    let struct_ident = item_struct.ident.clone();

    let align_val = layout.struct_align;

    // Check if struct contains GpuStruct (unknown-size) fields
    let has_unknown_size_fields = resolved_fields
        .iter()
        .any(|(_, gpu_type, _)| matches!(gpu_type, GpuType::GpuStruct { .. }));

    // Remove existing repr attributes
    item_struct
        .attrs
        .retain(|attr| !attr.path().is_ident("repr"));

    // Add #[repr(C, align(N))]
    let align_val_token = syn::LitInt::new(&align_val.to_string(), proc_macro2::Span::call_site());
    let repr_attr: syn::Attribute = syn::parse_quote!(#[repr(C, align(#align_val_token))]);
    item_struct.attrs.push(repr_attr);

    // Determine whether to inject explicit padding fields
    // Pod requires all bytes defined, so padding fields are mandatory for Pod
    let needs_explicit_padding = config.bytemuck || config.pad;

    // When bytemuck is on but we have unknown-size nested structs,
    // we can't guarantee no implicit padding, so Pod must be skipped
    let can_derive_pod = config.bytemuck && !has_unknown_size_fields;

    // Build derive list
    let derive_tokens = build_derive_tokens(item_struct, can_derive_pod);
    let derive_attr: syn::Attribute = syn::parse_quote!(#[derive(#(#derive_tokens),*)]);
    item_struct
        .attrs
        .retain(|attr| !attr.path().is_ident("derive"));
    item_struct.attrs.push(derive_attr);

    // Remove #[gpu_nested] field attributes
    if let syn::Fields::Named(fields) = &mut item_struct.fields {
        for field in &mut fields.named {
            field
                .attrs
                .retain(|attr| !(attr.path().segments.len() == 1 && attr.path().segments[0].ident == "gpu_nested"));
        }
    }

    // Transform bool fields to u32
    if let syn::Fields::Named(fields) = &mut item_struct.fields {
        for field in &mut fields.named {
            if let Some(ident) = &field.ident {
                if let Some((_, gpu_type, _)) = resolved_fields
                    .iter()
                    .find(|(name, _, _)| name == ident)
                {
                    if matches!(gpu_type, GpuType::Bool) {
                        field.ty = syn::parse_quote!(u32);
                    }
                }
            }
        }
    }

    // Insert explicit padding fields between fields and at tail
    // Only when bytemuck or pad is enabled, and no unknown-size nested fields
    if needs_explicit_padding && !has_unknown_size_fields {
        // Collect padding info before mutating
        let padding_gaps: Vec<(usize, usize)> = compute_padding_gaps(layout);
        let tail_padding = layout.tail_padding;

        inject_padding_fields(item_struct, &padding_gaps, tail_padding);
    }

    // Generate SIZE and ALIGN constants
    let size_val = layout.struct_size;
    let align_val_const = layout.struct_align;

    let size_align_tokens = if has_unknown_size_fields {
        quote! {
            /// Total size in bytes, matching GPU buffer stride.
            pub const SIZE: usize = core::mem::size_of::<Self>();
            /// Minimum alignment in bytes, matching GPU buffer binding requirements.
            pub const ALIGN: usize = core::mem::align_of::<Self>();
        }
    } else {
        quote! {
            /// Total size in bytes, matching GPU buffer stride.
            pub const SIZE: usize = #size_val;
            /// Minimum alignment in bytes, matching GPU buffer binding requirements.
            pub const ALIGN: usize = #align_val_const;
        }
    };

    // Generate offset constants if debug_layout or emit_offsets
    let offset_constants = if config.debug_layout || config.emit_offsets {
        let offset_consts: Vec<TokenStream2> = layout
            .fields
            .iter()
            .map(|field_layout| {
                let field_name = &field_layout.name;
                let offset_val = field_layout.offset;
                let const_ident = syn::Ident::new(
                    &format!("{}_OFFSET", field_name.to_string().to_uppercase()),
                    proc_macro2::Span::call_site(),
                );
                quote! {
                    #[doc = concat!("Byte offset of field `", stringify!(#field_name), "`")]
                    pub const #const_ident: usize = #offset_val;
                }
            })
            .collect();
        quote! { #(#offset_consts)* }
    } else {
        quote! {}
    };

    // Generate bool helper methods
    let bool_helpers = generate_bool_helpers(resolved_fields);

    // Generate compile-time assertions
    let assertions = generate_assertions(&struct_ident, layout, config, has_unknown_size_fields);

    let struct_tokens = quote! { #item_struct };

    let impl_tokens = quote! {
        impl #struct_ident {
            #size_align_tokens
            #offset_constants
            #bool_helpers
        }
    };

    quote! {
        #struct_tokens
        #impl_tokens
        #assertions
    }
}

/// Compute inter-field padding gaps from the layout.
/// Returns (field_index, gap_bytes) pairs.
fn compute_padding_gaps(layout: &StructLayout) -> Vec<(usize, usize)> {
    let mut gaps = Vec::new();
    let mut current_offset = 0usize;

    for (i, field_layout) in layout.fields.iter().enumerate() {
        if field_layout.offset > current_offset {
            gaps.push((i, field_layout.offset - current_offset));
        }
        current_offset = field_layout.offset + field_layout.size;
    }

    gaps
}

/// Insert explicit padding fields between existing fields and at the tail.
fn inject_padding_fields(
    item_struct: &mut ItemStruct,
    padding_gaps: &[(usize, usize)],
    tail_padding: usize,
) {
    if let syn::Fields::Named(fields) = &mut item_struct.fields {
        // Collect existing fields into a vec
        let original: Vec<syn::Field> = fields.named.iter().cloned().collect();
        fields.named.clear();

        let mut pad_counter = 0usize;
        let mut gap_iter = padding_gaps.iter().peekable();

        for (i, field) in original.into_iter().enumerate() {
            // Insert padding before this field if needed
            while let Some(&(gap_idx, gap_size)) = gap_iter.peek() {
                if *gap_idx == i {
                    let pad_ident = syn::Ident::new(
                        &format!("_prgpu_pad_{pad_counter}"),
                        proc_macro2::Span::call_site(),
                    );
                    pad_counter += 1;
                    let pad_field: syn::Field = syn::parse_quote!(
                        #[doc(hidden)]
                        #pad_ident: [u8; #gap_size]
                    );
                    fields.named.push(pad_field);
                    gap_iter.next();
                } else {
                    break;
                }
            }

            fields.named.push(field);
        }

        // Tail padding
        if tail_padding > 0 {
            let pad_ident = syn::Ident::new(
                "_prgpu_pad_tail",
                proc_macro2::Span::call_site(),
            );
            let pad_field: syn::Field = syn::parse_quote!(
                #[doc(hidden)]
                #pad_ident: [u8; #tail_padding]
            );
            fields.named.push(pad_field);
        }
    }
}

fn build_derive_tokens(item_struct: &ItemStruct, can_derive_pod: bool) -> Vec<proc_macro2::TokenStream> {
    let mut derives = Vec::new();

    let always = ["Clone", "Copy", "Debug"];
    for d in &always {
        let ident = syn::Ident::new(d, proc_macro2::Span::call_site());
        if !has_derive(item_struct, d) {
            derives.push(quote! { #ident });
        }
    }

    if can_derive_pod {
        if !has_derive(item_struct, "Pod") {
            derives.push(quote! { bytemuck::Pod });
        }
        if !has_derive(item_struct, "Zeroable") {
            derives.push(quote! { bytemuck::Zeroable });
        }
    }

    // Preserve existing user derives that aren't in our managed set
    for attr in &item_struct.attrs {
        if attr.path().is_ident("derive") {
            if let Ok(nested) = attr.parse_args_with(|input: syn::parse::ParseStream<'_>| {
                let mut idents = Vec::new();
                while !input.is_empty() {
                    let ident: syn::Ident = input.parse()?;
                    idents.push(ident.to_string());
                    if input.peek(syn::Token![,]) {
                        input.parse::<syn::Token![,]>()?;
                    }
                }
                Ok(idents)
            }) {
                for existing in nested {
                    if !always.contains(&existing.as_str())
                        && existing != "Pod"
                        && existing != "Zeroable"
                    {
                        let ident = syn::Ident::new(&existing, proc_macro2::Span::call_site());
                        derives.push(quote! { #ident });
                    }
                }
            }
        }
    }

    derives
}

fn has_derive(item_struct: &ItemStruct, name: &str) -> bool {
    for attr in &item_struct.attrs {
        if attr.path().is_ident("derive") {
            if let Ok(nested) = attr.parse_args_with(|input: syn::parse::ParseStream<'_>| {
                let mut idents = Vec::new();
                while !input.is_empty() {
                    let ident: syn::Ident = input.parse()?;
                    idents.push(ident.to_string());
                    if input.peek(syn::Token![,]) {
                        input.parse::<syn::Token![,]>()?;
                    }
                }
                Ok(idents)
            }) {
                if nested.iter().any(|s| s == name) {
                    return true;
                }
            }
        }
    }
    false
}

fn generate_bool_helpers(
    resolved_fields: &[(Ident, GpuType, proc_macro2::Span)],
) -> TokenStream2 {
    let helpers: Vec<TokenStream2> = resolved_fields
        .iter()
        .filter(|(_, gpu_type, _)| matches!(gpu_type, GpuType::Bool))
        .map(|(name, _, _)| {
            let helper_name = syn::Ident::new(
                &format!("{}_bool", name),
                proc_macro2::Span::call_site(),
            );
            quote! {
                #[doc = concat!("Returns `", stringify!(#name), "` as `bool` (mapped from u32 for GPU ABI).")]
                #[inline]
                pub fn #helper_name(&self) -> bool {
                    self.#name != 0
                }
            }
        })
        .collect();

    quote! { #(#helpers)* }
}

fn generate_assertions(
    struct_ident: &Ident,
    layout: &StructLayout,
    config: &GpuStructConfig,
    has_unknown_size_fields: bool,
) -> TokenStream2 {
    let size_val = layout.struct_size;
    let align_val = layout.struct_align;

    let mut assertions: Vec<TokenStream2> = Vec::new();

    if has_unknown_size_fields {
        assertions.push(quote! {
            assert!(<#struct_ident>::SIZE == core::mem::size_of::<#struct_ident>());
        });
        assertions.push(quote! {
            assert!(<#struct_ident>::ALIGN == core::mem::align_of::<#struct_ident>());
        });
    } else {
        assertions.push(quote! {
            assert!(core::mem::size_of::<#struct_ident>() == #size_val);
        });
        assertions.push(quote! {
            assert!(core::mem::align_of::<#struct_ident>() == #align_val);
        });
    }

    // Field offset assertions with debug_layout or emit_offsets
    if config.debug_layout || config.emit_offsets {
        for field_layout in &layout.fields {
            let field_name = &field_layout.name;
            let expected_offset = field_layout.offset;
            assertions.push(quote! {
                assert!(core::mem::offset_of!(#struct_ident, #field_name) == #expected_offset);
            });
        }
    }

    quote! {
        const _: () = {
            #(#assertions)*
        };
    }
}
