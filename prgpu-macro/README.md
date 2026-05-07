# prgpu-macro

Procedural macros backing [prgpu](https://crates.io/crates/prgpu). You don't add
this crate to your `Cargo.toml` directly — `prgpu` re-exports everything you
need.

## What's in here

- `#[gpu_struct]` — declarative struct-layout codegen:
  cross-target (Metal / CUDA / CPU) padding, explicit offset validation, and
  optional `bytemuck::{Pod, Zeroable}` derives. The macro guarantees that the
  emitted Rust struct byte-matches the layout expected by the Slang-generated
  shader code, so `#[repr(C)]` kernel params can round-trip through a
  `ConstantBuffer<T>` without hand-written padding.
- Related attribute parsing (`targets`, `align`, `allow_vec3`, `allow_bool`,
  `bytemuck`, `pad`, `strict`, `emit_offsets`, `debug_layout`) that let effect
  authors opt into stricter ABI guarantees or bypass them on a per-struct
  basis.

## Stability

`prgpu-macro` follows prgpu's release cadence. Pin your dependency on `prgpu`
only — its `Cargo.toml` already selects a compatible `prgpu-macro` version
transitively.

## License

Dual-licensed under MIT OR Apache-2.0.
