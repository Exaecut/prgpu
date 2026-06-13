use prgpu::gpu_struct;

#[gpu_struct]
pub struct Vec2 {
	pub x: f32,
	pub y: f32,
}

#[gpu_struct]
pub struct Vec3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub w: f32,
}

#[gpu_struct]
pub struct Transform {
	pub m: [[f32; 4]; 4],
}

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

#[gpu_struct]
pub struct MixedAlignment {
    pub x: u32,
    pub b: u8,
    pub y: u32,
}

#[test]
fn test_mixed_alignment() {
    // C repr: u32(4) + u8(1) + 3-byte pad + u32(4) = 12.
    assert_eq!(MixedAlignment::SIZE, core::mem::size_of::<MixedAlignment>());
    assert_eq!(MixedAlignment::ALIGN, core::mem::align_of::<MixedAlignment>());
}

#[gpu_struct(align = 16)]
pub struct AlignedStruct {
    pub val: f32,
}

#[test]
fn test_align_floor() {
    assert_eq!(AlignedStruct::ALIGN, 16);
    assert_eq!(core::mem::align_of::<AlignedStruct>(), 16);
    assert_eq!(AlignedStruct::SIZE, 16);
    assert_eq!(core::mem::size_of::<AlignedStruct>(), 16);
}

#[gpu_struct]
pub struct ArrayStruct {
    pub matrix: [[f32; 4]; 4],
    pub exposure: f32,
}

#[test]
fn test_array_struct() {
    assert_eq!(ArrayStruct::SIZE, core::mem::size_of::<ArrayStruct>());
    assert_eq!(ArrayStruct::ALIGN, core::mem::align_of::<ArrayStruct>());
    assert_eq!(ArrayStruct::SIZE, 68);
}

#[gpu_struct]
pub struct VectorFields {
    pub pos: Vec2,
    pub dir: Vec3,
}

#[test]
fn test_vector_fields() {
    assert_eq!(VectorFields::SIZE, core::mem::size_of::<VectorFields>());
    assert_eq!(VectorFields::ALIGN, core::mem::align_of::<VectorFields>());
    // Vec2(8) at offset 0, Vec3(16, align 16) at offset 16; total = 32.
    assert_eq!(VectorFields::SIZE, 32);
    assert_eq!(VectorFields::ALIGN, 16);
}

#[gpu_struct]
pub struct SingleField {
    pub value: f32,
}

#[test]
fn test_single_field() {
    assert_eq!(SingleField::SIZE, 4);
    assert_eq!(SingleField::ALIGN, 4);
}

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

#[gpu_struct]
pub struct BoolField {
    pub enabled: bool,
    pub value: f32,
}

#[test]
fn test_bool_field() {
    // bool becomes u32 (4 bytes), so size = 4 + 4 = 8.
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
    assert_eq!(DebugLayoutStruct::C_OFFSET, 8);
}

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

#[gpu_struct(targets(cuda, metal))]
pub struct CudaMetalStruct {
    pub x: f32,
}

#[test]
fn test_targets_attribute() {
    assert_eq!(CudaMetalStruct::SIZE, 4);
    assert_eq!(CudaMetalStruct::ALIGN, 4);
}

#[gpu_struct]
pub struct TransformField {
    pub transform: Transform,
}

#[test]
fn test_transform_field() {
    assert_eq!(TransformField::SIZE, core::mem::size_of::<TransformField>());
    assert_eq!(TransformField::ALIGN, core::mem::align_of::<TransformField>());
}

#[gpu_struct(bytemuck = false)]
pub struct NoBytemuckStruct {
    pub x: f32,
}

#[test]
fn test_no_bytemuck() {
    assert_eq!(NoBytemuckStruct::SIZE, 4);
    assert_eq!(NoBytemuckStruct::ALIGN, 4);
}

#[gpu_struct(allow_vec3)]
pub struct AllowVec3Struct {
    pub rgb: [f32; 3],
    pub alpha: f32,
}

#[test]
fn test_allow_vec3() {
    assert_eq!(AllowVec3Struct::SIZE, core::mem::size_of::<AllowVec3Struct>());
    assert_eq!(AllowVec3Struct::ALIGN, core::mem::align_of::<AllowVec3Struct>());
    // [f32; 3] = 12 bytes (align 4) + f32(4) = 16.
    assert_eq!(AllowVec3Struct::SIZE, 16);
}
