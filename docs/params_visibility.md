# `ParamApi` — visibility and actions

Adobe wants per-parameter `INVISIBLE` flags + AEGP `Hidden` dynamic
streams toggled on every `Cmd_UpdateParamsUi`. `ParamApi` lets effects
declare per-parameter rules once and have the adapter apply both surfaces.

## Declaration

```rust
fn ui(api: &mut ParamApi<Params>) -> Result<(), ae::Error> {
    api.visibility(|v| {
        v.show(Params::AllowOOBGlow,
               |_p, host| host.supports(Capability::FrameExpansion));

        v.show(Params::ExtensionOverride, |p, host| {
            use prgpu::params::CpuParams as _;
            host.supports(Capability::FrameExpansion)
                && p.checkbox(Params::AllowOOBGlow).unwrap_or(false)
        });

        v.show_all(
            [Params::TintBlendMode, Params::TintStrength, Params::TintColorA],
            |p, _host| {
                use prgpu::params::CpuParams as _;
                p.popup(Params::TintMode)
                 .map(|m| (m as u32).saturating_sub(1) != TINT_MODE_NONE)
                 .unwrap_or(false)
            });
    });

    api.actions(|a| {
        a.on_click(Params::ReloadShaders, |ctx| {
            ctx.hot_reload_shaders();
            Ok(())
        });
        a.on_click(Params::Feedback, |_ctx| {
            let _ = webbrowser::open("https://example.com/feedback");
            Ok(())
        });
    });
    Ok(())
}
```

## Visibility predicate signature

```rust
Fn(&Parameters<P>, HostCapabilities) -> bool + Send + Sync + 'static
```

- `&Parameters<P>` — read other parameters via the `prgpu::params::CpuParams`
  trait methods (`checkbox`, `popup`, `float`, `color`, `point`, `angle`).
- `HostCapabilities` — query `Capability::FrameExpansion`,
  `Capability::SourceOutputMayAlias`, etc., instead of `is_premiere()`
  checks.

Returning `true` shows the parameter; `false` hides it. The adapter
applies the AE PF flag + the AEGP dynamic-stream flag in the same step,
so authors don't write the AEGP `set_dynamic_stream_flag` plumbing.

## Action callback signature

```rust
Fn(&mut ActionContext) -> Result<(), &'static str> + Send + Sync + 'static
```

`ActionContext::hot_reload_shaders()` is the standard side-effect
callback. After the action returns, the adapter calls
`prgpu::gpu::pipeline::hot_reload()` if the flag is set.

For callbacks that need to do something else (open a URL, kick off a
licence retry), just call the function inside the closure. The closure
captures whatever it needs.

## Capability-driven visibility

| Capability                       | When supported                      |
|----------------------------------|-------------------------------------|
| `FrameExpansion`                 | After Effects only                  |
| `DynamicParamVisibility`         | always                              |
| `SourceOutputMayAlias`           | Premiere GPU                        |
| `NativePremiereGpuFilter`        | Premiere GPU                        |
| `SmartRenderGpu`                 | After Effects GPU path              |

Hide an effect's expansion / aliasing-aware controls on hosts that don't
support them with one rule:

```rust
v.show(Params::AllowOOBGlow, |_p, host| host.supports(Capability::FrameExpansion));
```

## When `ui` is called

The adapter calls `Effect::ui` on every `Cmd_UpdateParamsUi`. Predicates
re-evaluate against live parameter values, so toggling a checkbox the
visibility rule depends on takes effect on the next UI tick.

The same `ui` rules also drive `Cmd_UserChangedParam`: when the user
clicks a parameter, the adapter looks up the matching `on_click` callback
in the cached rule set and invokes it.

## Don't put setup in `ui`

`ui` runs every UI tick. `Effect::params` runs once at `Cmd_ParamsSetup`.
Adobe `Parameters::add_*` is positional and not idempotent — calling it
twice re-adds the parameter with a new index. Keep `add_*` calls in
`Effect::params` only.
