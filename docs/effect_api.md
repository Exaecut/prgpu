# `Effect` trait — the high-level public API

Effect crates implement one trait. The Adobe adapters drive every PF / GPU
selector against it. No per-effect `handle_command`, no per-effect
`pr::GpuFilter::render`, no manual `Configuration` mutation per pass.

```rust
pub trait Effect: Sized + Default + Send + Sync + 'static {
    type Params: SetupParams + Eq + Hash + Copy + Debug + Send + Sync + 'static;
    type FrameData: Copy + Send + Sync + 'static;
    type License: LicenseGate;

    fn descriptor() -> EffectDescriptor;

    fn params(p: &mut Parameters<Self::Params>, in_data: InData, out_data: OutData)
        -> Result<(), ae::Error>;

    fn ui(api: &mut ParamApi<Self::Params>) -> Result<(), ae::Error> { Ok(()) }

    fn frame_data(ctx: FrameDataContext<Self::Params>)
        -> Result<Self::FrameData, ae::Error>;

    fn expansion(ctx: ExpansionContext<Self::Params>)
        -> Result<ExpansionExtent, ae::Error> { Ok(ExpansionExtent::none()) }

    fn pipeline(g: &mut RenderGraph<Self::FrameData>);
}
```

## Required associated types

| Type        | Constraint                                                | What it is |
|-------------|-----------------------------------------------------------|------------|
| `Params`    | `SetupParams + Eq + Hash + Copy + Debug + Send + Sync`    | Your effect's `enum Params { ... }` |
| `FrameData` | `Copy + Send + Sync + 'static`                            | Per-frame snapshot of resolved kernel params + any host-derived state |
| `License`   | `LicenseGate`                                             | `NoLicenseGate` if your effect ships unlocked |

## Required methods

### `descriptor()`

Static metadata Adobe needs at registration. Display name, version,
options-button label, premiere pixel formats.

```rust
fn descriptor() -> EffectDescriptor {
    EffectDescriptor::new("My Effect")
        .about("My Effect — short description shown in About dialog")
        .version(env!("CARGO_PKG_VERSION"))
        .options_button("Infos")
        .premiere_pixel_formats([
            ae::pr::PixelFormat::Bgra4444_32f,
            ae::pr::PixelFormat::Bgra4444_8u,
        ])
}
```

### `params(p, in_data, out_data)`

Adobe parameter setup. Called once at `Cmd_ParamsSetup`. Idiomatic:
delegate to your `enum Params` `SetupParams::setup` impl.

### `frame_data(ctx)`

Per-frame extractor. The adapter calls this once per render selector with
a `FrameDataContext` that abstracts AE PF vs Premiere GPU host state.
`ctx.float(P)`, `ctx.popup_zero_based(P)`, `ctx.checkbox(P)`,
`ctx.color(P)`, `ctx.point_pct(P)` all work the same regardless of host.

`kernel_params!` generates a `from_context` constructor on the params
struct so you rarely need to touch the extractors directly:

```rust
fn frame_data(ctx: FrameDataContext<Params>) -> Result<FrameData, ae::Error> {
    Ok(FrameData {
        prefilter: PrefilterParams::from_context(&ctx)?,
        upsample:  UpsampleParams::from_context(&ctx)?,
        composite: CompositeParams::from_context(&ctx)?,
        quality:   ctx.popup_zero_based(Params::Quality)?,
        frame_index: ctx.frame_index(),
        time_seconds: ctx.time_seconds(),
    })
}
```

### `pipeline(g)`

Declares the render graph once per effect-instance lifetime. The adapter
caches the resulting graph and replays it against each frame's `FrameData`.
See [`render_graph.md`](render_graph.md) for the full pass DSL.

## Optional methods

### `ui(api)`

Called every `Cmd_UpdateParamsUi`. Declares per-parameter visibility
predicates and click-action handlers. The adapter applies AE PF
`INVISIBLE` flags + AEGP `DynamicStreamFlags::Hidden` from the same
declaration, so authors don't write the visibility plumbing twice.

```rust
fn ui(api: &mut ParamApi<Params>) -> Result<(), ae::Error> {
    api.visibility(|v| {
        v.show(Params::AllowOOBGlow,
               |_p, host| host.supports(Capability::FrameExpansion));
    });
    api.actions(|a| {
        a.on_click(Params::ReloadShaders, |ctx| {
            ctx.hot_reload_shaders();
            Ok(())
        });
    });
    Ok(())
}
```

### `expansion(ctx)`

Per-side pixel inflation for SmartPreRender. Returning a non-zero extent
makes the adapter expand `result_rect` / `max_result_rect` and set
`RETURNS_EXTRA_PIXELS`. Default: no expansion.

Premiere disables expansion via `Capability::FrameExpansion`; effects
should gate on the capability rather than `is_premiere()` directly.

## Hooking into the Adobe binding macros

The adapter implements `params_setup` / `handle_command` as **inherent**
methods (not as a trait, because `AdobePluginGlobal` is generated locally
inside each effect crate by `ae::define_effect!`). The user writes a
3-line trampoline:

```rust
#[derive(Default)]
struct Plugin(prgpu::adobe::ae::EffectAdapter<MyEffect>);

impl AdobePluginGlobal for Plugin {
    fn params_setup(&self, params: &mut Parameters<Params>, in_data: InData, out_data: OutData)
        -> Result<(), ae::Error>
    { self.0.params_setup(params, in_data, out_data) }

    fn handle_command(&mut self, command: Command, in_data: InData, out_data: OutData,
                      params: &mut Parameters<Params>) -> Result<(), ae::Error>
    { self.0.handle_command(command, in_data, out_data, params) }
}

ae::define_effect!(Plugin, (), Params);

pub type PremiereGPU = prgpu::adobe::premiere::GpuFilterAdapter<MyEffect>;
premiere::define_gpu_filter!(PremiereGPU);
```

The Premiere adapter implements `pr::GpuFilter` directly and needs no
trampoline.

## See also

- [`render_graph.md`](render_graph.md) — graph DSL, mip chains, source policy
- [`kernel_descriptor.md`](kernel_descriptor.md) — `Kernel<P>` + `declare_kernel!`
- [`adobe_adapters.md`](adobe_adapters.md) — adapter pair lifecycle
- [`license_gate.md`](license_gate.md) — opt-in licence checks
- [`migration_high_level_api.md`](migration_high_level_api.md) — old → new mapping
