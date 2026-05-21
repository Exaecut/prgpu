# `Kernel<P>` + evolved `declare_kernel!`

`declare_kernel!(name, P)` emits a per-kernel module containing every
entry the graph executor needs (shader bytes, entry-point name, CPU
dispatch fns, GPU dispatch fn, CPU render adapter, `Kernel<P>`
constructor) plus deprecated top-level wrappers for backward compat.

```rust
declare_kernel!(bloom_prefilter, BloomPrefilterParams);
```

generates:

```rust
pub mod bloom_prefilter {
    pub const SHADER_SRC: &[u8];                      // metallib (Metal) or PTX (CUDA)
    pub const ENTRY_POINT: &str;                      // = "bloom_prefilter"
    pub const CPU_DISPATCH: CpuDispatchFn;
    pub const CPU_DISPATCH_TILE: CpuDispatchTileFn;

    pub unsafe fn gpu(cfg, params) -> Result<(), &'static str>;
    pub fn cpu(in_data, in_layer, out_layer, cfg, params) -> Result<(), ae::Error>;

    pub fn kernel() -> Kernel<BloomPrefilterParams>;
}

#[deprecated] pub unsafe fn bloom_prefilter(cfg, params) -> Result<(), &'static str>;
#[deprecated] pub fn bloom_prefilter_cpu(in_data, in_layer, out_layer, cfg, params) -> Result<(), ae::Error>;
#[deprecated] pub const BLOOM_PREFILTER_CPU_DISPATCH: CpuDispatchFn;
#[deprecated] pub const BLOOM_PREFILTER_CPU_DISPATCH_TILE: CpuDispatchTileFn;
```

The deprecated top-level forms keep older effects compiling. `mod foo` and
`fn foo()` coexist because they live in separate Rust namespaces (type vs
value).

## `Kernel<P>`

```rust
pub struct Kernel<P: KernelParams> { /* opaque */ }

impl<P: KernelParams> Kernel<P> {
    pub const fn name(&self) -> &'static str;
    pub const fn shader_src(&self) -> &'static [u8];
    pub const fn entry_point(&self) -> &'static str;

    pub unsafe fn dispatch_gpu(&self, cfg: &Configuration, params: P)
        -> Result<(), &'static str>;

    pub fn dispatch_cpu(&self, in_data: &ae::InData, in_layer: &ae::Layer,
                        out_layer: &mut ae::Layer, cfg: &Configuration, params: P)
        -> Result<(), ae::Error>;

    pub unsafe fn dispatch_cpu_direct(&self, cfg: &Configuration, params: P);
}
```

`dispatch_cpu_direct` is the AE-host-free path used by the graph executor
for resource→resource passes. It uses the rayon tile dispatcher directly
without an `ae::Layer::iterate_with` fast-path branch.

## `KernelParams`

```rust
pub trait KernelParams: Copy + Sync + Sized + 'static {
    const SIZE: usize;
    const ALIGN: usize;
}
```

`kernel_params! { ... }` auto-implements this trait. Manually-written
constant-buffer structs should `#[gpu_struct]`-annotate the type and
implement `KernelParams` explicitly:

```rust
#[gpu_struct]
pub struct MyParams { pub x: f32, pub y: f32 }

impl KernelParams for MyParams {
    const SIZE: usize = Self::SIZE;
    const ALIGN: usize = Self::ALIGN;
}
```

The `Sync` bound is required by the CPU dispatcher (rayon worker threads
share a raw pointer to the params buffer). Every `gpu_struct`-laid-out
type satisfies this trivially — only scalar / fixed-array fields are
allowed, all of which are `Sync`.

## Binding contract

The 5-buffer Metal / CUDA binding the dispatcher hardcodes:

| Slot | Buffer            | Slang signature                    |
|------|-------------------|------------------------------------|
| 0    | outgoing (read)   | `StructuredBuffer<uint> outgoing`  |
| 1    | incoming (read)   | `StructuredBuffer<uint> incoming`  |
| 2    | dest (read/write) | `RWStructuredBuffer<uint> dst`     |
| 3    | frame             | `ConstantBuffer<FrameParams> frame`|
| 4    | params            | `ConstantBuffer<MyParams>      params`|

Every Slang shader must declare all five even when it ignores some — the
dispatcher always binds `setBuffer atIndex: N` for `N ∈ 0..5`. Verify by
reading the generated `target/debug/build/<crate>-*/out/<kernel>_bindings.rs`
file: `METAL_<kernel>_PARAM_COUNT` should be 5.

## See also

- [`config_builder.md`](config_builder.md) — `Configuration` + `ConfigBuilder`
- [`render_graph.md`](render_graph.md) — how the graph dispatches a `Kernel<P>`
