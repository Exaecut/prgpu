# `InvocationBase` + `ConfigBuilder`

Adobe hands every render path a different bag of host objects. The
adapters extract those into a single `InvocationBase` before the graph
executor or any per-pass code runs. `ConfigBuilder` then turns the base
+ a few per-pass slot bindings into a `Configuration` (the low-level ABI
struct kernel dispatchers consume).

## `Configuration` (low-level ABI)

Stays as the sole struct passed to generated `kernel::gpu(cfg, params)` /
`kernel::cpu(...)` calls. Don't mutate it field-by-field in effect code —
that's what `ConfigBuilder` is for.

## `InvocationBase` (per-render normalised state)

```rust
pub struct InvocationBase {
    pub host: Host,                    // AfterEffects | Premiere
    pub backend: Backend,              // Cpu | Cuda | Metal | OpenCL
    pub render_kind: RenderKind,
    pub device_handle: *mut c_void,
    pub context_handle: Option<*mut c_void>,
    pub command_queue_handle: *mut c_void,
    pub bytes_per_pixel: u32,
    pub pixel_layout: PixelLayout,     // Rgba | Bgra | Vuya601 | Vuya709
    pub time: f32,
    pub progress: f32,
    pub render_generation: u64,
    pub main_source: FrameBinding,
    pub incoming_source: Option<FrameBinding>,
    pub outgoing_source: Option<FrameBinding>,
    pub output: FrameBinding,
}
```

`InvocationBase` is built ONCE per render call. The adapter:

- AE PF CPU path: `EffectAdapter::build_invocation_cpu`
- AE PF GPU path: `EffectAdapter::build_invocation_gpu`
- Premiere GPU path: `GpuFilterAdapter::build_invocation`

Effect code never constructs one directly outside of tests.

## `FrameBinding` (typed pixel-buffer view)

```rust
pub struct FrameBinding {
    pub data: *mut c_void,
    pub pitch_px: i32,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub bytes_per_pixel: u32,
    pub pixel_layout: PixelLayout,
}
```

`unsafe impl Send + Sync` — the raw pointer is a host-owned token that
survives the dispatch. Same contract as `Configuration`.

## `ConfigBuilder` (per-pass `Configuration` assembly)

```rust
let cfg = ConfigBuilder::new(&base)
    .source(PassBinding::MainSource)         // slot 0 (outgoing)
    .input(PassBinding::Inline(bloom_l0))    // slot 1 (incoming) — optional
    .target(PassBinding::Output)             // slot 2 (dest)
    .dispatch_size(out_w, out_h)             // dispatch grid
    .mip_levels(5)                           // optional
    .build()?;
```

| Slot method   | Maps to                  |
|---------------|--------------------------|
| `source(b)`   | `outgoing` (slot 0)      |
| `outgoing(b)` | same (alias)             |
| `input(b)`    | `incoming` (slot 1)      |
| `incoming(b)` | same (alias)             |
| `target(b)`   | `dest` (slot 2)          |
| `dest(b)`     | same (alias)             |

`PassBinding` variants:

- `MainSource` / `Output` / `OutgoingSource` / `IncomingSource` — resolve
  through `InvocationBase`.
- `Inline(FrameBinding)` — pre-built binding (snapshots, mip slices).
- `Null` — explicit no-op.

When `incoming` is unset, it mirrors `outgoing` — matches kernels that
ignore slot 1 and avoids a separate `dispatch_size` argument for it.

`build()` validates: missing target → `MissingDest`, zero dispatch size →
`ZeroDispatchSize`.

## When to use it directly

The graph executor does this for you. Reach for `ConfigBuilder` directly
only when:

- writing a unit test that needs a synthetic `Configuration`,
- writing custom code that bypasses the graph (e.g. a one-off
  `prepare_source_copy` step).

For everything else, declare passes through
[`render_graph.md`](render_graph.md).
