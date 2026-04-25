use crate::parse::GpuTarget;
use crate::types::GpuType;
use syn::Ident;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FieldLayout {
    pub name: Ident,
    pub gpu_type: GpuType,
    pub offset: usize,
    pub size: usize,
    pub alignment: usize,
}

#[derive(Debug, Clone)]
pub struct StructLayout {
    pub fields: Vec<FieldLayout>,
    pub struct_size: usize,
    pub struct_align: usize,
    pub tail_padding: usize,
}

fn align_up(offset: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

pub fn compute_layout(
    fields: &[(Ident, GpuType)],
    align_floor: Option<usize>,
    _targets: &[GpuTarget],
) -> StructLayout {
    let mut field_layouts = Vec::with_capacity(fields.len());

    // Determine maximum field alignment
    let max_field_align = fields
        .iter()
        .map(|(_, gpu_type)| {
            let a = gpu_type.alignment();
            if a == 0 { 1 } else { a }
        })
        .max()
        .unwrap_or(1);

    let align_floor_val = align_floor.unwrap_or(1);
    let struct_align = max_field_align.max(align_floor_val);

    let mut current_offset = 0usize;

    for (name, gpu_type) in fields {
        let field_align = gpu_type.alignment();
        let field_align = if field_align == 0 { 1 } else { field_align };
        let field_size = gpu_type.size();

        let field_offset = align_up(current_offset, field_align);
        current_offset = field_offset + field_size;

        field_layouts.push(FieldLayout {
            name: name.clone(),
            gpu_type: gpu_type.clone(),
            offset: field_offset,
            size: field_size,
            alignment: field_align,
        });
    }

    let struct_size = align_up(current_offset, struct_align);
    let tail_padding = struct_size - current_offset;

    StructLayout {
        fields: field_layouts,
        struct_size,
        struct_align,
        tail_padding,
    }
}
