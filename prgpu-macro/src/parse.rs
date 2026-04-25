use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Result, Token};

#[derive(Debug, Clone)]
pub struct GpuStructConfig {
    pub targets: Vec<GpuTarget>,
    pub align: Option<usize>,
    pub allow_vec3: bool,
    pub allow_bool: bool,
    pub debug_layout: bool,
    pub emit_offsets: bool,
    pub strict: bool,
    pub bytemuck: bool,
    pub pad: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTarget {
    Cuda,
    Metal,
    OpenCL,
}

impl Default for GpuStructConfig {
    fn default() -> Self {
        Self {
            targets: vec![GpuTarget::Cuda, GpuTarget::Metal],
            align: None,
            allow_vec3: false,
            allow_bool: true,
            debug_layout: false,
            emit_offsets: false,
            strict: false,
            bytemuck: true,
            pad: false,
        }
    }
}

struct GpuStructConfigParser {
    config: GpuStructConfig,
}

impl Parse for GpuStructConfigParser {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut config = GpuStructConfig::default();

        while !input.is_empty() {
            let key: Ident = input.parse()?;

            match key.to_string().as_str() {
                "targets" => {
                    let content;
                    syn::parenthesized!(content in input);
                    let mut targets = Vec::new();
                    while !content.is_empty() {
                        let target: Ident = content.parse()?;
                        match target.to_string().as_str() {
                            "cuda" => targets.push(GpuTarget::Cuda),
                            "metal" => targets.push(GpuTarget::Metal),
                            "opencl" => targets.push(GpuTarget::OpenCL),
                            other => {
                                return Err(syn::Error::new(
                                    target.span(),
                                    format!("unknown GPU target '{other}'; valid: cuda, metal, opencl"),
                                ));
                            }
                        }
                        if content.peek(Token![,]) {
                            content.parse::<Token![,]>()?;
                        }
                    }
                    if targets.is_empty() {
                        return Err(syn::Error::new(key.span(), "targets list cannot be empty"));
                    }
                    config.targets = targets;
                }
                "align" => {
                    input.parse::<Token![=]>()?;
                    let lit: LitInt = input.parse()?;
                    let val = lit.base10_parse::<usize>()?;
                    if val == 0 || (val & (val - 1)) != 0 {
                        return Err(syn::Error::new(
                            lit.span(),
                            "align must be a positive power of 2",
                        ));
                    }
                    config.align = Some(val);
                }
                "allow_vec3" => {
                    config.allow_vec3 = true;
                }
                "allow_bool" => {
                    config.allow_bool = true;
                }
                "debug_layout" => {
                    config.debug_layout = true;
                }
                "emit_offsets" => {
                    config.emit_offsets = true;
                }
                "strict" => {
                    config.strict = true;
                }
                "bytemuck" => {
                    input.parse::<Token![=]>()?;
                    let val: syn::LitBool = input.parse()?;
                    config.bytemuck = val.value;
                }
                "pad" => {
                    config.pad = true;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown attribute '{other}'; valid: targets, align, allow_vec3, allow_bool, debug_layout, emit_offsets, strict, bytemuck, pad"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self { config })
    }
}

pub fn parse_config(tokens: &TokenStream) -> Result<GpuStructConfig> {
    if tokens.is_empty() {
        return Ok(GpuStructConfig::default());
    }

    let parser: GpuStructConfigParser = syn::parse2(tokens.clone())?;
    Ok(parser.config)
}
