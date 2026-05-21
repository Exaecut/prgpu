# Adobe adapters

`prgpu::adobe::ae::EffectAdapter<E>` and
`prgpu::adobe::premiere::GpuFilterAdapter<E>` bridge the `Effect` trait
into the host-specific traits the existing prgpu macros expect.

## After Effects PF — `EffectAdapter<E>`

`AdobePluginGlobal` is generated locally inside each effect crate by
`ae::define_effect!`, so `EffectAdapter<E>` exposes its selectors as
inherent methods and the user writes a 3-line trampoline:

```rust
#[derive(Default)]
struct Plugin(prgpu::adobe::ae::EffectAdapter<MyEffect>);

impl AdobePluginGlobal for Plugin {
    fn params_setup(&self, params: &mut Parameters<MyParams>, in_data: InData, out_data: OutData)
        -> Result<(), ae::Error>
    { self.0.params_setup(params, in_data, out_data) }

    fn handle_command(&mut self, command: Command, in_data: InData, out_data: OutData,
                      params: &mut Parameters<MyParams>) -> Result<(), ae::Error>
    { self.0.handle_command(command, in_data, out_data, params) }
}

ae::define_effect!(Plugin, (), MyParams);
```

### Selector mapping

| AE selector             | Effect method / behaviour                           |
|-------------------------|------------------------------------------------------|
| `Cmd_GlobalSetup`       | install logger; `descriptor()` pixel formats; `License::initialize` |
| `Cmd_About`             | `descriptor().about_text` + version                  |
| `Cmd_ParamsSetup`       | `Effect::params`                                     |
| `Cmd_UpdateParamsUi`    | `Effect::ui` collects rules; adapter applies AE PF + AEGP visibility |
| `Cmd_UserChangedParam`  | matches against cached `ParamApi::actions` rules; hot-reloads shaders if requested |
| `Cmd_FrameSetup`        | `Effect::expansion` + `Effect::frame_data`           |
| `Cmd_FrameSetdown`      | drops cached `FrameData`                             |
| `Cmd_Render`            | builds CPU `InvocationBase`; runs graph              |
| `Cmd_SmartPreRender`    | `Effect::expansion` → result/max rect + `RETURNS_EXTRA_PIXELS` |
| `Cmd_SmartRender`       | builds CPU `InvocationBase`; runs graph              |
| `Cmd_SmartRenderGpu`    | builds GPU `InvocationBase`; runs graph              |
| `Cmd_GpuDeviceSetup`    | sets `SupportsGpuRenderF32` for `Metal` / `Cuda`     |

Licence checks gate `Cmd_FrameSetup` / `Cmd_Render` / `Cmd_SmartRender*`
through `Effect::License::is_valid`.

## Premiere GPU — `GpuFilterAdapter<E>`

`pr::GpuFilter` is a real public trait; the adapter implements it
directly. Effects use the type alias as-is:

```rust
pub type PremiereGPU = prgpu::adobe::premiere::GpuFilterAdapter<MyEffect>;
premiere::define_gpu_filter!(PremiereGPU);
```

### `pr::GpuFilter` mapping

| Method                      | Behaviour                                              |
|-----------------------------|--------------------------------------------------------|
| `global_init`               | no-op                                                  |
| `global_destroy`            | `pipeline::cleanup` + `gpu::buffer::cleanup`           |
| `get_frame_dependencies`    | returns `Err(None)` (no dependencies)                  |
| `precompute`                | no-op                                                  |
| `render`                    | builds `InvocationBase` from `GPURenderProperties` + `Configuration::effect`; calls `Effect::frame_data`; runs graph |

Premiere-specific quirks the adapter handles automatically:

- **PPix dimension override** — when source/dest PPix bounds disagree
  with `render_params.render_*()` (Premiere 25.2 native-res), prefer them.
- **Pitch validation** — drops frames where source pitch < expected
  destination pitch.
- **Source-output aliasing** — handled via `Capability::SourceOutputMayAlias`
  + the graph's `SourcePolicy` (see [`source_snapshot.md`](source_snapshot.md)).

## Why two adapters?

The two host SDKs have different lifecycle models:

- AE PF is a single global plugin instance with command-selector dispatch.
- Premiere GPU is a per-effect-type filter implementing one trait method
  per phase.

The adapters factor each into a thin layer over the shared `Effect` trait
+ shared graph executor. Per-effect code stops branching on host.

## Raw access

Effects that need to reach into raw Adobe state (e.g. for an unusual
`Cmd_About` payload, custom GPU device-handle inspection, or telemetry)
can wrap `EffectAdapter<E>` instead of using it as the type alias and
intercept selectors before delegating. Future versions of prgpu may add an
explicit `Effect::hooks` API for this; for now, the wrap-and-delegate
pattern is the recommended escape hatch.
