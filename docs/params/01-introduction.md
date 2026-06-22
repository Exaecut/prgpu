# `params!` — declarative effect parameter UI

A prgpu effect's UI is a single `prgpu::params! { ... }` block. You declare
the parameters, their kinds, their groups, and (optionally) the routes that
switch whole panels in and out. The macro generates everything the Adobe
adapters need to register, snapshot, and drive those parameters on both
After Effects and Premiere Pro — no hand-written `PF_Cmd_PARAMS_SETUP`,
no per-host checkout boilerplate, no `ParamFlag` plumbing.

This guide is the high-level entry point. For worked scenarios see
[`02-use-cases.md`](02-use-cases.md); for the full attribute reference see
[`03-reference.md`](03-reference.md).

## What the DSL is for

Adobe effects expose their controls as a flat, ordered list of host
parameters. The host binds saved projects to that **registration order** —
reordering shipped parameters breaks every project that used the effect.
Writing the registration by hand means repeating the same value once for
setup, again for the CPU snapshot, again for the GPU snapshot, and again
for visibility — across two hosts with different quirks (1-based AE popups,
pixel-space points, 8-bit colour, Premiere popups arriving as `Float32`).

`params!` replaces all of that with one declaration. You write each
parameter **once**, in the order it must ship. The macro derives:

- the `Params` discriminant enum (one variant per parameter, in order),
- one zero-sized **marker** per parameter for typed reads,
- a host-agnostic `Snapshot` (every host quirk resolved once, at snapshot
  time),
- the registration code for both adapters,
- an optional `Router` enum + hidden route store when you tag groups with
  `route = ...`,
- the button click/action dispatch table,
- the static-text label bindings.

The read side is one typed call: `ctx.get(Strength) -> f32`,
`ctx.get(Tint) -> Color`, `ctx.get(Quality) -> Quality`. No locks, no map
lookups — it compiles to an array index plus a match.

## Where it fits

An effect crate is five declarations in a pipeline:

```
prgpu::params!    →  prgpu::kernel!  →  impl Effect  →  prgpu::register_effect!  →  build.rs
   UI + snapshot      GPU param struct    render graph      Adobe FFI wiring       PiPL + shaders
```

`params!` is the first one. It produces the `Params` type that the `Effect`
trait, the `kernel!` block, and `register_effect!` all reference by name.

## Setup

`params!` lives in the `prgpu` crate, re-exported from the root. Add it to
your effect's `Cargo.toml`:

```toml
[dependencies]
prgpu = { path = "../prgpu" }
```

The macro and the full runtime surface come in through the prelude. Put
this at the top of `src/params.rs`:

```rust
use prgpu::prelude::*;
```

`prgpu::params!` itself is a top-level re-export of the proc macro, so no
extra path is needed to invoke it.

## Getting started

The declaration is a Rust enum whose variants are your parameters, each
tagged with one attribute that says what kind of control it is. Create
`src/params.rs`:

```rust
use prgpu::prelude::*;

prgpu::params! {
    pub enum Params {
        #[slider(label = "Strength", range = 0.0..=100.0, default = 50.0, percent)]
        Strength,
        #[checkbox(label = "Invert", default = false)]
        Invert,
        #[button(label = "Feedback", on_click = open_feedback)]
        Feedback,
    }
}

fn open_feedback() {
    let _ = webbrowser::open("https://example.com/feedback");
}
```

That single block generates:

- `enum Params { Strength = 1, Invert, Feedback }` (discriminants start at
  `1`; `0` is reserved for the AE input layer),
- zero-sized markers `struct Strength;` `struct Invert;` `struct Feedback;`,
- a `Snapshot` storing one normalized value per parameter,
- the `ParamsSpec` implementation that registers the three controls on both
  hosts and snapshots them per frame,
- the button dispatch table wiring `Feedback` → `open_feedback`.

### Reading values in the pipeline

The pipeline and frame closures receive a `&Ctx<Params>`. Each marker's
`Value` is inferred, so reads are fully typed:

```rust
fn pipeline(g: &mut Graph<Params>) {
    g.pass(k::myeffect::kernel())
        .params(|ctx| MyEffectParams {
            strength: ctx.get(Strength) / 100.0, // percent slider → 0..1
            invert:   ctx.get(Invert),
        });
}
```

| Declaration                 | Read                          | Type    |
|-----------------------------|-------------------------------|---------|
| `#[slider(...)]`            | `ctx.get(Strength)`           | `f32`   |
| `#[angle(...)]`             | `ctx.get(Angle)`              | `f32`   |
| `#[checkbox(...)]`          | `ctx.get(Invert)`             | `bool`  |
| `#[color(...)]`             | `ctx.get(Tint)`               | `Color` |
| `#[point(...)]`             | `ctx.get(Anchor)`             | `Point2`|
| `#[popup(options = [...])]` | `ctx.get(Mode)`               | `u32`   |
| `#[popup(options = Enum)]`  | `ctx.get(Qual)`               | `Enum`  |
| `#[blend_mode(...)]`        | `ctx.get(Blend)`              | `BlendMode` |

`Color` channels and `Point2` coordinates arrive already normalized to
`0..1`; popups arrive **0-based** regardless of host. You never branch on
the host in your read code.

### Wiring it into the effect

`src/lib.rs` names the generated enum as the effect's `Params` type and
registers it:

```rust
use prgpu::prelude::*;
use crate::{kernel as k, params::*};

#[derive(Default)]
pub struct MyEffect;

impl Effect for MyEffect {
    type Params = Params;
    fn pipeline(g: &mut Graph<Params>) {
        g.pass(k::myeffect::kernel());
    }
}

prgpu::register_effect!(MyEffect);
```

`register_effect!` generates the AE plug-in entry points and the Premiere
GPU filter adapter, driving every selector through the `Effect` trait.
See [`effect_api.md`](../effect_api.md) for the full trait.

## Rules of thumb

- **Declaration order is sacred.** After Effects binds saved projects to
  registration order. Only **append** new parameters to a shipped effect;
  never reorder, rename, or delete a shipped variant. Group markers are
  injected automatically — keep variants in the order you want them shown.
- **One kind attribute per variant.** Every variant needs exactly one of
  `slider`, `checkbox`, `color`, `angle`, `point`, `popup`, `blend_mode`,
  `button`, `layer`, `custom`, or `label`. `group` is separate and frames a
  range of variants.
- **`label` is required** for every kind except `custom` and `label`. The
  macro errors at compile time if it is missing.
- **`default` must lie inside `range`.** A slider with `default = 120.0,
  range = 0.0..=100.0` is a compile error, not a runtime clamp.
- **Groups must be closed.** Every `#[group("...")]` needs a matching
  `#[group(end)]`; the macro errors on an unclosed group.
- **`debug_only` hides in release.** Tag a control with `debug_only` and it
  is collapsed + invisible in release builds. The single
  `#[checkbox(debug_only)]` param (conventionally named `Debug`) is wired
  as the effect's debug-view switch.

## What to read next

- [`02-use-cases.md`](02-use-cases.md) — grouped panels, typed enum popups,
  routed wizards, wide-range sliders, secondary inputs, custom params.
- [`03-reference.md`](03-reference.md) — every attribute, key, generated
  item, and the `Router` / `ActionCtx` APIs, with example snippets.
- [`effect_api.md`](../effect_api.md) — the `Effect` trait the generated
  `Params` plugs into.
- [`params_visibility.md`](../params_visibility.md) — dynamic
  show/hide and label rules in `Effect::ui`.
