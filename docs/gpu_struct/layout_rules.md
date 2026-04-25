# GPU Layout Rules

## Alignment Rules

The `#[gpu_struct]` macro computes struct layout following GPU ABI conventions. These rules differ subtly from C's `#[repr(C)]` alone because GPU architectures have stricter alignment requirements.

### Scalar Alignment

| Type | Size (bytes) | Alignment (bytes) |
|------|-------------|-------------------|
| `u8`, `i8` | 1 | 1 |
| `u16`, `i16` | 2 | 2 |
| `u32`, `i32`, `f32` | 4 | 4 |
| `u64`, `i64`, `f64` | 8 | 8 |
| `bool` (as `u32`) | 4 | 4 |

### Vector Alignment

GPU vector types have alignment equal to their size (padded):

| Type | Size (bytes) | Alignment (bytes) |
|------|-------------|-------------------|
| `Vec2` | 8 | 8 |
| `Vec3` | 16 | 16 |

`Vec3` is 16 bytes (not 12) because GPU APIs (CUDA, Metal) align `float3` to 16 bytes with an implicit padding `w` component. The prgpu `Vec3` struct includes an explicit `_pad: u32` field to match this.

### Array Alignment

Arrays inherit the alignment of their element type:

```rust
// [f32; 4] has alignment 4 (same as f32)
// [[f32; 4]; 4] has alignment 4 (same as f32)
```

### Struct Alignment

A struct's alignment is the maximum alignment of all its fields, rounded up to the `align = N` floor if specified:

```rust
// struct { a: u32, b: Vec3 } → max(4, 16) = 16
// struct { a: f32 } with align = 16 → max(4, 16) = 16
```

## Padding Rules

### Inter-Field Padding

Fields are laid out in declaration order. Each field starts at the next offset that satisfies its alignment:

```
struct Example {
    a: u32,    // offset 0, size 4
    b: u8,     // offset 4, size 1
               // 3 bytes padding (offsets 5-7)
    c: u32,    // offset 8, size 4
}
// Total: 12 bytes, alignment 4
```

### Tail Padding

The struct size is rounded up to its alignment:

```
struct WithVec3 {
    v: Vec3,   // offset 0, size 16
    x: f32,    // offset 16, size 4
}
// Total before rounding: 20, alignment 16
// Rounded up: 32 bytes
```

### Explicit Padding with bytemuck

When `bytemuck = true` (default), the macro injects explicit padding fields so that `Pod` can be derived. The `Pod` trait requires that every byte of the struct is part of a field — no implicit padding allowed.

The injected fields are named `_prgpu_pad_0`, `_prgpu_pad_1`, etc. for inter-field gaps, and `_prgpu_pad_tail` for tail padding. They are marked `#[doc(hidden)]` to avoid cluttering documentation.

## CUDA vs Metal Differences

Both CUDA and Metal follow the same basic alignment rules for scalar and vector types. The key differences:

| Aspect | CUDA | Metal |
|--------|------|-------|
| `float3` alignment | 16 bytes | 16 bytes |
| `float4` alignment | 16 bytes | 16 bytes |
| Struct size rounding | To max alignment | To max alignment |
| `bool` representation | 4 bytes (`int`) | 4 bytes (`int32_t`) |

OpenCL may differ in some edge cases, but is not yet a supported target.

## Offset Computation Algorithm

The macro uses this algorithm to compute layout:

```
current_offset = 0
for each field:
    field_align = gpu_alignment(field.type)
    field_offset = align_up(current_offset, field_align)
    field.size = gpu_size(field.type)
    current_offset = field_offset + field.size

struct_align = max(all field alignments, config.align)
struct_size = align_up(current_offset, struct_align)
tail_padding = struct_size - current_offset
```

Where `align_up(offset, align) = (offset + align - 1) & !(align - 1)`.

## Why Not Just `#[repr(C)]`?

`#[repr(C)]` alone is not sufficient for GPU ABI compatibility because:

1. **Rust's `bool` is 1 byte**, but GPU `bool`/`int` is 4 bytes. The macro handles this mapping.
2. **Vec3 needs 16-byte alignment**, which `#[repr(C)]` won't enforce unless you add `align(16)` manually.
3. **bytemuck::Pod requires explicit padding**, not the implicit padding that `#[repr(C)]` provides.
4. **Compile-time verification** — the macro asserts that the generated struct actually has the expected size and alignment.

## Common Pitfalls

### Forgetting Vec3 alignment

```rust
// BAD: Vec3 at offset 8, violating 16-byte alignment
#[repr(C)]
struct Bad {
    x: f32,    // offset 0
    v: Vec3,   // offset 4 — WRONG, should be 16
}

// GOOD: macro places Vec3 at offset 16
#[gpu_struct]
struct Good {
    x: f32,    // offset 0
    v: Vec3,   // offset 16 — correct
}
```

### Using [f32; 3] instead of Vec3

```rust
// BAD: [f32; 3] is 12 bytes with alignment 4 — doesn't match GPU float3
#[gpu_struct]  // ERROR: [f32; 3] rejected by default
struct Bad {
    rgb: [f32; 3],
}

// GOOD: use Vec3 for GPU-compatible 16-byte float3
#[gpu_struct]
struct Good {
    rgb: prgpu::Vec3,
}
```

### Nested structs without #[gpu_nested]

```rust
// BAD: unknown nested struct
#[gpu_struct]
struct Bad {
    inner: MyOtherStruct,  // ERROR: not recognized as GPU-safe
}

// GOOD: annotate with #[gpu_nested]
#[gpu_struct]
struct Good {
    #[gpu_nested]
    inner: MyOtherStruct,
}
```
