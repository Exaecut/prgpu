use prgpu::gpu_struct;

// Test 1: Basic scalar struct with no padding needed
#[gpu_struct]
pub struct Scalars {
    pub a: u32,
    pub b: f32,
    pub c: u32,
}

#[test]
fn test_scalars_size_and_align() {
    assert_eq!(Scalars::SIZE, 12);
    assert_eq!(Scalars::ALIGN, 4);
    assert_eq!(core::mem::size_of::<Scalars>(), 12);
    assert_eq!(core::mem::align_of::<Scalars>(), 4);
}

// Test 2: Struct requiring inter-field padding (u8 after u32)
#[gpu_struct]
pub struct MixedAlignment {
    pub x: u32,
    pub b: u8,
    pub y: u32,
}

#[test]
fn test_mixed_alignment() {
    // u32(4) + pad(3) + u8(1) + u32(4) = 12, with C repr it's:
    // offset 0: u32(4), offset 4: u8(1), offset 5-7: pad, offset 8: u32(4) = 12
    assert_eq!(MixedAlignment::SIZE, core::mem::size_of::<MixedAlignment>());
    assert_eq!(MixedAlignment::ALIGN, core::mem::align_of::<MixedAlignment>());
}

// Test 3: align = 16 forcing alignment floor
#[gpu_struct(align = 16)]
pub struct AlignedStruct {
    pub val: f32,
}

#[test]
fn test_align_floor() {
    assert_eq!(AlignedStruct::ALIGN, 16);
    assert_eq!(core::mem::align_of::<AlignedStruct>(), 16);
    // Size should be 16 due to align(16)
    assert_eq!(AlignedStruct::SIZE, 16);
    assert_eq!(core::mem::size_of::<AlignedStruct>(), 16);
}

// Test 4: Array fields including nested arrays
#[gpu_struct]
pub struct ArrayStruct {
    pub matrix: [[f32; 4]; 4],
    pub exposure: f32,
}

#[test]
fn test_array_struct() {
    assert_eq!(ArrayStruct::SIZE, core::mem::size_of::<ArrayStruct>());
    assert_eq!(ArrayStruct::ALIGN, core::mem::align_of::<ArrayStruct>());
    // [[f32;4];4] = 64 bytes, f32 = 4 bytes, total with padding = 68
    // But align is 4, so 68 is already aligned
    assert_eq!(ArrayStruct::SIZE, 68);
}

// Test 5: Vec2 and Vec3 as field types
#[gpu_struct]
pub struct VectorFields {
    pub pos: prgpu::Vec2,
    pub dir: prgpu::Vec3,
}

#[test]
fn test_vector_fields() {
    assert_eq!(VectorFields::SIZE, core::mem::size_of::<VectorFields>());
    assert_eq!(VectorFields::ALIGN, core::mem::align_of::<VectorFields>());
    // Vec2(8) at offset 0, Vec3(16, align 16) at offset 16, total = 32
    assert_eq!(VectorFields::SIZE, 32);
    assert_eq!(VectorFields::ALIGN, 16);
}

// Test 6: Single field struct
#[gpu_struct]
pub struct SingleField {
    pub value: f32,
}

#[test]
fn test_single_field() {
    assert_eq!(SingleField::SIZE, 4);
    assert_eq!(SingleField::ALIGN, 4);
}

// Test 7: Default attributes (empty #[gpu_struct])
#[gpu_struct]
pub struct DefaultAttrs {
    pub x: u64,
    pub y: u32,
}

#[test]
fn test_default_attrs() {
    assert_eq!(DefaultAttrs::SIZE, core::mem::size_of::<DefaultAttrs>());
    assert_eq!(DefaultAttrs::ALIGN, core::mem::align_of::<DefaultAttrs>());
}

// Test 8: bool field (transformed to u32 by default)
#[gpu_struct]
pub struct BoolField {
    pub enabled: bool,
    pub value: f32,
}

#[test]
fn test_bool_field() {
    // bool becomes u32 (4 bytes), so size = 4 + 4 = 8
    assert_eq!(BoolField::SIZE, 8);
    assert_eq!(BoolField::ALIGN, 4);
    assert_eq!(core::mem::size_of::<BoolField>(), 8);
}

#[test]
fn test_bool_helper() {
    let b = BoolField { enabled: 1u32, value: 1.0 };
    assert!(b.enabled_bool());
    let b2 = BoolField { enabled: 0u32, value: 0.0 };
    assert!(!b2.enabled_bool());
}

// Test 9: Nested struct with #[gpu_nested]
#[gpu_struct]
pub struct InnerGpu {
    pub x: f32,
    pub y: f32,
}

#[gpu_struct]
pub struct OuterGpu {
    #[gpu_nested]
    pub inner: InnerGpu,
    pub z: f32,
}

#[test]
fn test_nested_struct() {
    assert_eq!(OuterGpu::SIZE, core::mem::size_of::<OuterGpu>());
    assert_eq!(OuterGpu::ALIGN, core::mem::align_of::<OuterGpu>());
}

// Test 10: debug_layout generates offset constants
#[gpu_struct(debug_layout)]
pub struct DebugLayoutStruct {
    pub a: u32,
    pub b: f32,
    pub c: u8,
}

#[test]
fn test_debug_layout_offsets() {
    assert_eq!(DebugLayoutStruct::A_OFFSET, 0);
    assert_eq!(DebugLayoutStruct::B_OFFSET, 4);
    // c comes after b at offset 8, but alignment is 1 so no padding before it
    assert_eq!(DebugLayoutStruct::C_OFFSET, 8);
}

// Test 11: pad attribute injects explicit padding
#[gpu_struct(pad, align = 16)]
pub struct PaddedStruct {
    pub val: f32,
}

#[test]
fn test_padded_struct() {
    assert_eq!(PaddedStruct::SIZE, 16);
    assert_eq!(PaddedStruct::ALIGN, 16);
    assert_eq!(core::mem::size_of::<PaddedStruct>(), 16);
}

// Test 12: targets attribute
#[gpu_struct(targets(cuda, metal))]
pub struct CudaMetalStruct {
    pub x: f32,
}

#[test]
fn test_targets_attribute() {
    assert_eq!(CudaMetalStruct::SIZE, 4);
    assert_eq!(CudaMetalStruct::ALIGN, 4);
}

// Test 13: Transform as built-in nested type
#[gpu_struct]
pub struct TransformField {
    pub transform: prgpu::Transform,
}

#[test]
fn test_transform_field() {
    assert_eq!(TransformField::SIZE, core::mem::size_of::<TransformField>());
    assert_eq!(TransformField::ALIGN, core::mem::align_of::<TransformField>());
}

// Test 14: bytemuck = false opt-out
#[gpu_struct(bytemuck = false)]
pub struct NoBytemuckStruct {
    pub x: f32,
}

#[test]
fn test_no_bytemuck() {
    assert_eq!(NoBytemuckStruct::SIZE, 4);
    assert_eq!(NoBytemuckStruct::ALIGN, 4);
}

// Test 15: allow_vec3 attribute
#[gpu_struct(allow_vec3)]
pub struct AllowVec3Struct {
    pub rgb: [f32; 3],
    pub alpha: f32,
}

#[test]
fn test_allow_vec3() {
    assert_eq!(AllowVec3Struct::SIZE, core::mem::size_of::<AllowVec3Struct>());
    assert_eq!(AllowVec3Struct::ALIGN, core::mem::align_of::<AllowVec3Struct>());
    // [f32;3] = 12 bytes (align 4), f32 = 4 bytes, total = 16
    assert_eq!(AllowVec3Struct::SIZE, 16);
}
