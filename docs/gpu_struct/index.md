# `#[gpu_struct]` — GPU-Safe Struct Macro

The `#[gpu_struct]` attribute macro transforms ordinary Rust structs into GPU-ABI-compatible host-side representations, ensuring correct memory layout for CUDA, Metal, and future OpenCL kernel parameter passing.

## Quick Start

```rust
use prgpu::gpu_struct;

#[gpu_struct]
pub struct MyParams {
    pub time: f32,
    pub intensity: f32,
    pub enabled: bool,   // automatically mapped to u32 for GPU ABI
}
```

This generates:

- `#[repr(C, align(N))]` with the correct GPU alignment
- `bytemuck::Pod` + `bytemuck::Zeroable` derives (when possible)
- `Clone`, `Copy`, `Debug` derives
- `SIZE` and `ALIGN` constants on the impl block
- `enabled_bool()` helper method for the `bool → u32` field
- Compile-time const assertions verifying size and alignment
- Explicit padding fields (when `bytemuck` or `pad` is enabled) so every byte is defined

## Attribute Options

| Option | Default | Description |
|--------|---------|-------------|
| `targets(cuda, metal)` | `cuda, metal` | Target GPU backends for layout rules |
| `align = N` | auto | Force minimum alignment to N bytes (must be power of 2) |
| `bytemuck = bool` | `true` | Auto-derive `bytemuck::Pod` + `bytemuck::Zeroable` |
| `pad` | `false` | Inject explicit padding fields even without bytemuck |
| `allow_vec3` | `false` | Allow `[f32; 3]`, `[u32; 3]`, `[i32; 3]` arrays |
| `allow_bool` | `true` | Allow `bool` fields (mapped to `u32`) |
| `debug_layout` | `false` | Emit `FIELD_OFFSET` constants and field offset assertions |
| `emit_offsets` | `false` | Emit `FIELD_OFFSET` constants (same as debug_layout but without assertions) |
| `strict` | `false` | Reject `bool`, `f64`, and other questionable types |

## Supported Field Types

### Scalars

| Rust type | GPU size | GPU align |
|-----------|----------|-----------|
| `u8`, `i8` | 1 | 1 |
| `u16`, `i16` | 2 | 2 |
| `u32`, `i32`, `f32` | 4 | 4 |
| `u64`, `i64`, `f64` | 8 | 8 |
| `bool` | 4 (as `u32`) | 4 |

### Vector Types

| Type | Size | Align | Notes |
|------|------|-------|-------|
| `Vec2` | 8 | 8 | `{ x: f32, y: f32 }` |
| `Vec3` | 16 | 16 | `{ x: f32, y: f32, z: f32, _pad: u32 }` |

`Vec2` and `Vec3` are recognized only from trusted paths: `Vec2`, `crate::Vec2`, `crate::types::Vec2`, `crate::types::maths::Vec2`, `prgpu::Vec2`, `prgpu::types::Vec2`, `prgpu::types::maths::Vec2` (and the same for `Vec3`).

### Arrays

Fixed-size arrays of supported types: `[f32; 4]`, `[[f32; 4]; 4]`, etc.

By default, `[f32; 3]`, `[u32; 3]`, and `[i32; 3]` are **rejected** because their GPU ABI is ambiguous (3×4 = 12 bytes without padding vs. Vec3's 16 bytes with padding). Use `Vec3` instead, or enable `allow_vec3` if you truly want the unpadded 12-byte layout.

### Nested Structs

Other `#[gpu_struct]`-annotated structs and built-in types like `Transform` are allowed as nested fields, but you must annotate them with `#[gpu_nested]`:

```rust
#[gpu_struct]
pub struct Inner {
    pub x: f32,
    pub y: f32,
}

#[gpu_struct]
pub struct Outer {
    #[gpu_nested]
    pub inner: Inner,
    pub z: f32,
}
```

When a struct contains `#[gpu_nested]` fields, `bytemuck::Pod` cannot be derived automatically (the macro can't verify the nested type has no implicit padding), so only `Clone`, `Copy`, `Debug` are derived.

## Bool → u32 Mapping

GPU kernels don't have a standard `bool` type. By default, `#[gpu_struct]` maps `bool` fields to `u32` on the host side:

```rust
#[gpu_struct]
pub struct Params {
    pub enabled: bool,  // becomes `enabled: u32` in the transformed struct
}

// Access the bool value:
let p = Params { enabled: 1u32 };
assert!(p.enabled_bool());  // true

let p2 = Params { enabled: 0u32 };
assert!(!p2.enabled_bool()); // false
```

In `strict` mode, `bool` is rejected entirely — use `u32` explicitly.

## Rejected Types

| Type | Reason |
|------|--------|
| `usize`, `isize` | Platform-dependent size (32 or 64 bit) |
| `Vec<T>`, `Box<T>`, etc. | Heap-allocated, not GPU-safe |
| `&T`, `&mut T` | References don't make sense in GPU memory |
| Tuples `(A, B)` | No stable ABI layout |
| `other::Vec3` | Not a trusted path — use `prgpu::Vec3` |

## Compile-Time Guarantees

The macro emits const assertions that fail at compile time if the generated struct doesn't match the expected layout:

```rust
const _: () = {
    assert!(core::mem::size_of::<MyStruct>() == EXPECTED);
    assert!(core::mem::align_of::<MyStruct>() == EXPECTED);
};
```

With `debug_layout`, field offset assertions are also emitted:

```rust
const _: () = {
    assert!(core::mem::offset_of!(MyStruct, field) == EXPECTED_OFFSET);
};
```

## Padding Field Injection

When `bytemuck = true` (default) or `pad` is enabled, the macro injects explicit padding fields so that `bytemuck::Pod` can be derived (Pod requires every byte of the struct to be defined):

```rust
#[gpu_struct]
pub struct MixedAlignment {
    pub x: u32,      // offset 0, size 4
    pub b: u8,       // offset 4, size 1
    pub y: u32,      // offset 8, size 4 (3 bytes padding at offset 5-7)
}

// Transformed to:
#[repr(C, align(4))]
pub struct MixedAlignment {
    pub x: u32,
    pub b: u8,
    #[doc(hidden)]
    _prgpu_pad_0: [u8; 3],
    pub y: u32,
    #[doc(hidden)]
    _prgpu_pad_tail: [u8; 0],  // no tail padding needed here
}
```

When `bytemuck = false` and `pad` is not set, no padding fields are injected — the struct relies on `#[repr(C)]`'s implicit padding. This is simpler but means `bytemuck::Pod` cannot be derived.

## Interaction with `kernel_params!`

The `#[gpu_struct]` macro is designed to coexist with the existing `kernel_params!` macro. Over time, `kernel_params!` will be updated to recognize `#[gpu_struct]`-annotated types and use their `SIZE`/`ALIGN` constants directly.
