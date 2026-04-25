use crate::parse::GpuStructConfig;
use syn::spanned::Spanned;
use syn::Type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    Bool,
    Vec2,
    Vec3,
    Array {
        element: Box<GpuType>,
        count: usize,
    },
    GpuStruct {
        name: String,
    },
    #[allow(dead_code)]
    Unknown,
}

impl GpuType {
    pub fn size(&self) -> usize {
        match self {
            GpuType::U8 | GpuType::I8 => 1,
            GpuType::U16 | GpuType::I16 => 2,
            GpuType::U32 | GpuType::I32 | GpuType::F32 | GpuType::Bool => 4,
            GpuType::U64 | GpuType::I64 | GpuType::F64 => 8,
            GpuType::Vec2 => 8,
            GpuType::Vec3 => 16,
            GpuType::Array { element, count } => {
                let elem_size = element.size();
                let elem_align = element.alignment();
                let elem_stride = align_up(elem_size, elem_align);
                elem_stride * count
            }
            GpuType::GpuStruct { .. } => 0,
            GpuType::Unknown => 0,
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            GpuType::U8 | GpuType::I8 => 1,
            GpuType::U16 | GpuType::I16 => 2,
            GpuType::U32 | GpuType::I32 | GpuType::F32 | GpuType::Bool => 4,
            GpuType::U64 | GpuType::I64 | GpuType::F64 => 8,
            GpuType::Vec2 => 8,
            GpuType::Vec3 => 16,
            GpuType::Array { element, .. } => element.alignment(),
            GpuType::GpuStruct { .. } => 0,
            GpuType::Unknown => 0,
        }
    }
}

fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

// Trusted path prefixes for Vec2/Vec3 recognition.
const TRUSTED_VEC2_PATHS: &[&str] = &[
    "Vec2",
    "crate::Vec2",
    "crate::types::Vec2",
    "crate::types::maths::Vec2",
    "prgpu::Vec2",
    "prgpu::types::Vec2",
    "prgpu::types::maths::Vec2",
];

const TRUSTED_VEC3_PATHS: &[&str] = &[
    "Vec3",
    "crate::Vec3",
    "crate::types::Vec3",
    "crate::types::maths::Vec3",
    "prgpu::Vec3",
    "prgpu::types::Vec3",
    "prgpu::types::maths::Vec3",
];

// Built-in GPU-safe struct types always approved as nested fields.
const BUILTIN_GPU_STRUCTS: &[&str] = &["Transform"];

fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn is_trusted_path(path_str: &str, trusted: &[&str]) -> bool {
    trusted.iter().any(|t| path_str == *t)
}

/// Resolve a `syn::Type` to a `GpuType`.
///
/// `is_gpu_nested`: whether the field has a `#[gpu_nested]` attribute,
/// indicating the user asserts this nested struct type is ABI-safe.
pub fn resolve_type(
    ty: &Type,
    config: &GpuStructConfig,
    is_gpu_nested: bool,
) -> Result<GpuType, syn::Error> {
    match ty {
        Type::Path(type_path) => {
            // Reject generic types (e.g. Vec<u8>, Option<T>)
            if let Some(last_seg) = type_path.path.segments.last() {
                if !matches!(last_seg.arguments, syn::PathArguments::None) {
                    return Err(syn::Error::new(
                        ty.span(),
                        "generic types are not supported in #[gpu_struct]; \
                         use concrete GPU-compatible types only",
                    ));
                }
            }

            let path_str = path_to_string(&type_path.path);
            let final_segment = type_path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();

            // Scalar types
            match final_segment.as_str() {
                "u8" => return Ok(GpuType::U8),
                "i8" => return Ok(GpuType::I8),
                "u16" => return Ok(GpuType::U16),
                "i16" => return Ok(GpuType::I16),
                "u32" => return Ok(GpuType::U32),
                "i32" => return Ok(GpuType::I32),
                "f32" => return Ok(GpuType::F32),
                "u64" => return Ok(GpuType::U64),
                "i64" => return Ok(GpuType::I64),
                "f64" => {
                    if config.strict {
                        return Err(syn::Error::new(
                            ty.span(),
                            "f64 is rejected in strict mode; \
                             GPU kernels typically use f32 for performance and compatibility",
                        ));
                    }
                    return Ok(GpuType::F64);
                }
                "bool" => {
                    if config.strict {
                        return Err(syn::Error::new(
                            ty.span(),
                            "bool is rejected in strict mode; use u32 instead",
                        ));
                    }
                    return Ok(GpuType::Bool);
                }
                "usize" => {
                    return Err(syn::Error::new(
                        ty.span(),
                        "usize is platform-dependent (32 or 64 bit); use u32 or u64 for GPU ABI",
                    ));
                }
                "isize" => {
                    return Err(syn::Error::new(
                        ty.span(),
                        "isize is platform-dependent (32 or 64 bit); use i32 or i64 for GPU ABI",
                    ));
                }
                _ => {}
            }

            // Trusted Vec2/Vec3 recognition
            if final_segment == "Vec2" {
                if is_trusted_path(&path_str, TRUSTED_VEC2_PATHS) {
                    return Ok(GpuType::Vec2);
                }
                return Err(syn::Error::new(
                    ty.span(),
                    format!(
                        "type `{path_str}` is not a recognized GPU Vec2; \
                         only Vec2 / crate::Vec2 / prgpu::Vec2 are trusted. \
                         Import from prgpu::types or use a different name."
                    ),
                ));
            }

            if final_segment == "Vec3" {
                if is_trusted_path(&path_str, TRUSTED_VEC3_PATHS) {
                    return Ok(GpuType::Vec3);
                }
                return Err(syn::Error::new(
                    ty.span(),
                    format!(
                        "type `{path_str}` is not a recognized GPU Vec3; \
                         only Vec3 / crate::Vec3 / prgpu::Vec3 are trusted. \
                         Import from prgpu::types or use a different name."
                    ),
                ));
            }

            // Built-in GPU-safe structs
            if BUILTIN_GPU_STRUCTS.contains(&final_segment.as_str()) {
                return Ok(GpuType::GpuStruct {
                    name: final_segment,
                });
            }

            // User-approved nested struct via #[gpu_nested]
            if is_gpu_nested {
                return Ok(GpuType::GpuStruct {
                    name: final_segment,
                });
            }

            // Unknown struct type without approval
            Err(syn::Error::new(
                ty.span(),
                format!(
                    "nested struct `{final_segment}` is not recognized as GPU-safe; \
                     annotate the field with #[gpu_nested] to assert ABI safety, \
                     or replace with an approved type"
                ),
            ))
        }

        Type::Array(type_array) => {
            let elem_ty = &*type_array.elem;
            let gpu_elem = resolve_type(elem_ty, config, is_gpu_nested)?;

            let count = extract_array_len(&type_array)?;

            // Check for vec3 problem: [f32; 3], [u32; 3], [i32; 3]
            if count == 3 && !config.allow_vec3 {
                if matches!(gpu_elem, GpuType::F32) {
                    return Err(syn::Error::new(
                        ty.span(),
                        "[f32; 3] is rejected by default (ambiguous GPU ABI layout); \
                         use Vec3, [f32; 4], or enable allow_vec3",
                    ));
                }
                if matches!(gpu_elem, GpuType::U32) {
                    return Err(syn::Error::new(
                        ty.span(),
                        "[u32; 3] is rejected by default (ambiguous GPU ABI layout); \
                         use [u32; 4] or enable allow_vec3",
                    ));
                }
                if matches!(gpu_elem, GpuType::I32) {
                    return Err(syn::Error::new(
                        ty.span(),
                        "[i32; 3] is rejected by default (ambiguous GPU ABI layout); \
                         use [i32; 4] or enable allow_vec3",
                    ));
                }
            }

            Ok(GpuType::Array {
                element: Box::new(gpu_elem),
                count,
            })
        }

        Type::Tuple(_) => {
            Err(syn::Error::new(
                ty.span(),
                "tuple types are not supported in #[gpu_struct]; use a named struct or array",
            ))
        }

        Type::Reference(_) => {
            Err(syn::Error::new(
                ty.span(),
                "reference types are not supported in #[gpu_struct]; \
                 GPU structs must be Copy and self-contained",
            ))
        }

        Type::Ptr(_) => {
            Err(syn::Error::new(
                ty.span(),
                "pointer types are not supported in #[gpu_struct]; \
                 GPU structs must be Copy and self-contained",
            ))
        }

        _ => Err(syn::Error::new(
            ty.span(),
            "unsupported type in #[gpu_struct]; \
             use scalar, array, or approved GPU struct types",
        )),
    }
}

fn extract_array_len(type_array: &syn::TypeArray) -> Result<usize, syn::Error> {
    let expr = &type_array.len;
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(lit_int),
            ..
        }) => lit_int
            .base10_parse::<usize>()
            .map_err(|e| syn::Error::new(expr.span(), format!("invalid array size: {e}"))),
        _ => Err(syn::Error::new(
            expr.span(),
            "only literal array sizes are supported in #[gpu_struct]",
        )),
    }
}
