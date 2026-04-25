use crate::parse::GpuStructConfig;
use syn::ItemStruct;

/// Validate existing repr attributes on the struct for conflicts with #[gpu_struct].
pub fn validate_repr(item_struct: &ItemStruct, config: &GpuStructConfig) -> Result<(), syn::Error> {
    for attr in &item_struct.attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }

        let Ok(nested) = attr.parse_args_with(|input: syn::parse::ParseStream<'_>| {
            let mut tokens = Vec::new();
            while !input.is_empty() {
                let tt: proc_macro2::TokenTree = input.parse()?;
                tokens.push(tt);
            }
            Ok(tokens)
        }) else {
            continue;
        };

        // Check for repr(packed)
        for tt in &nested {
            if let proc_macro2::TokenTree::Ident(ident) = tt {
                if ident == "packed" {
                    return Err(syn::Error::new(
                        ident.span(),
                        "#[repr(packed)] is incompatible with #[gpu_struct]; \
                         GPU structs require natural alignment for correct memory layout",
                    ));
                }

                if ident == "align" {
                    // Check if existing align conflicts with config.align
                    if let Some(config_align) = config.align {
                        // Parse the align value from existing repr
                        let tokens_str = nested
                            .iter()
                            .map(|t| t.to_string())
                            .collect::<String>();
                        // Format: align(N) or align = N
                        if let Some(paren_start) = tokens_str.find('(') {
                            if let Some(paren_end) = tokens_str.find(')') {
                                let val_str = &tokens_str[paren_start + 1..paren_end];
                                if let Ok(existing_align) = val_str.trim().parse::<usize>() {
                                    if existing_align != config_align {
                                        return Err(syn::Error::new(
                                            ident.span(),
                                            format!(
                                                "conflicting alignment: \
                                                 #[repr(C, align({existing_align}))] conflicts \
                                                 with align = {config_align}; \
                                                 remove the existing repr or adjust align"
                                            ),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
