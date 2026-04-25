# `#[gpu_struct]` Examples

## Basic Kernel Parameters

```rust
use prgpu::gpu_struct;

#[gpu_struct]
pub struct BlurParams {
    pub radius: f32,
    pub direction: prgpu::Vec2,
    pub quality: u32,
}
```

Generated constants:

```rust
assert_eq!(BlurParams::SIZE, 16);  // Vec2(8) at align 8, radius(4) before, quality(4) after
assert_eq!(BlurParams::ALIGN, 8);
```

## Forced Alignment

For structs passed to APIs requiring specific alignment (e.g., Metal buffer bindings at 16-byte alignment):

```rust
#[gpu_struct(align = 16)]
pub struct AlignedParams {
    pub time: f32,
    pub delta: f32,
}
```

```rust
assert_eq!(AlignedParams::SIZE, 16);
assert_eq!(AlignedParams::ALIGN, 16);
```

## Bool Fields

```rust
#[gpu_struct]
pub struct ToggleParams {
    pub enabled: bool,    // stored as u32
    pub strength: f32,
}

// Construct with u32 values:
let params = ToggleParams { enabled: 1u32, strength: 0.5 };

// Read as bool:
if params.enabled_bool() {
    // ...
}
```

## Debug Layout

```rust
#[gpu_struct(debug_layout)]
pub struct DebugParams {
    pub a: u32,
    pub b: u8,
    pub c: u32,
}

// Offset constants are generated:
assert_eq!(DebugParams::A_OFFSET, 0);
assert_eq!(DebugParams::B_OFFSET, 4);
assert_eq!(DebugParams::C_OFFSET, 8);
```

## Nested GPU Structs

```rust
#[gpu_struct]
pub struct TransformParams {
    pub position: prgpu::Vec2,
    pub scale: prgpu::Vec2,
    pub rotation: f32,
}

#[gpu_struct]
pub struct EffectParams {
    #[gpu_nested]
    pub transform: TransformParams,
    pub opacity: f32,
}
```

Note: When a struct contains `#[gpu_nested]` fields, `Pod` is not auto-derived because the macro can't verify the nested type's internal padding at compile time.

## Opting Out of Bytemuck

```rust
#[gpu_struct(bytemuck = false)]
pub struct NoPodParams {
    pub x: f32,
    pub y: f32,
}
```

No `Pod`/`Zeroable` derives, no explicit padding fields. Simpler struct definition but you lose `bytemuck::cast_slice()` etc.

## Allowing [f32; 3]

By default, `[f32; 3]` is rejected because its 12-byte size conflicts with Vec3's 16-byte GPU layout. If you specifically want the unpadded 12-byte layout:

```rust
#[gpu_struct(allow_vec3)]
pub struct RgbParams {
    pub rgb: [f32; 3],
    pub alpha: f32,
}
```

```rust
assert_eq!(RgbParams::SIZE, 16);
```

## Strict Mode

```rust
#[gpu_struct(strict)]
pub struct StrictParams {
    pub x: f32,
    // pub flag: bool,  // ERROR: bool rejected in strict mode
    // pub precise: f64, // ERROR: f64 rejected in strict mode
}
```

## Matrix Type (Nested Arrays)

```rust
#[gpu_struct]
pub struct MatrixParams {
    pub mvp: [[f32; 4]; 4],
    pub exposure: f32,
}
```

```rust
assert_eq!(MatrixParams::SIZE, 68);  // 64 + 4
assert_eq!(MatrixParams::ALIGN, 4);
```

## Transform Field

`Transform` is a built-in GPU-safe type:

```rust
#[gpu_struct]
pub struct SceneParams {
    pub transform: prgpu::Transform,
    pub opacity: f32,
}
```
