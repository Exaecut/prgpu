# `params!` — use cases

Worked scenarios showing the common shapes the DSL supports. Every snippet
is a self-contained example you can adapt to your own effect. For the
attribute grammar see [`03-reference.md`](03-reference.md); for the
orientation see [`01-introduction.md`](01-introduction.md).

## 1. A compact effect with one group

The minimal shape: a feedback button, a debug checkbox, and a single
grouped pair of controls. Groups are just visual folding in the ECW — no
routing, no visibility rules.

```rust
prgpu::params! {
    pub enum Params {
        #[button(label = "Feedback", on_click = open_feedback)]
        Feedback,
        #[checkbox(label = "Debug", default = false, debug_only)]
        Debug,

        #[group("Tint")]
        #[color(label = "Tint Color", default = "#000000")]
        Tint,
        #[slider(label = "Strength", range = 0.0..=100.0, default = 50.0, percent, precision = 3)]
        Strength,
        #[group(end)]
    }
}

fn open_feedback() {
    let _ = webbrowser::open("https://example.com/feedback");
}
```

`Debug` carries `debug_only`, so it is hidden in release builds and becomes
the effect's debug-view switch. `Strength` is a percent slider — the value
still reads as `0.0..=100.0` via `ctx.get(Strength)`; the `percent` flag only
sets the host's display formatting, so keep any `/ 100.0` in your pipeline
expression.

## 2. A panel effect with several groups

When the control set is large, organize it into folded groups. Each
`#[group("Title")] ... #[group(end)]` range becomes a collapsible section.
Parameters between groups (or before the first group) sit in the top level
of the panel.

```rust
prgpu::params! {
    pub enum Params {
        #[button(label = "Feedback", on_click = open_feedback)]
        Feedback,
        #[checkbox(label = "Debug", default = false, debug_only)]
        Debug,

        #[group("Source")]
        #[point(label = "Center", default = (50.0, 50.0))]
        Center,
        #[slider(label = "Size", range = 0.0..=100.0, default = 5.0, percent, precision = 2)]
        Size,
        #[slider(label = "Threshold", range = 0.0..=100.0, default = 35.0, percent, precision = 2)]
        Threshold,
        #[color(label = "Key Color", default = "#FFFFFF")]
        KeyColor,
        #[group(end)]

        #[group("Output")]
        #[slider(label = "Length", range = 0.0..=100.0, default = 100.0)]
        Length,
        #[slider(label = "Spread", range = 0.0..=360.0, default = 0.0, precision = 1)]
        Spread,
        #[group(end)]

        #[group("Coloring")]
        #[popup(label = "Color Mode", options = ["Solid", "Gradient", "Rainbow"], default = 0)]
        ColorMode,
        #[color(label = "Tint Color", default = "#FFFFFF")]
        TintColor,
        #[blend_mode(label = "Blend Mode", default = Add)]
        BlendMode,
        #[group(end)]

        #[group("Quality")]
        #[slider(label = "Samples", range = 2.0..=1024.0, default = 64.0, precision = 0)]
        Samples,
        #[popup(label = "Falloff", options = ["Linear", "Exponential", "Inverse Square"], default = 1)]
        Falloff,
        #[group(end)]

        #[checkbox(label = "Chromatic Aberration", default = false)]
        CAEnable,
        #[slider(label = "CA Amount", range = 0.0..=100.0, default = 10.0, percent, precision = 2)]
        CAAmount,
    }
}
```

Notes on this shape:

- `#[blend_mode(default = Add)]` is sugar for a popup over
  `prgpu::BlendMode`. It reads back as a typed `BlendMode`:
  `ctx.get(BlendMode) -> BlendMode`.
- `#[point(default = (50.0, 50.0))]` is in **layer pixel space** on AE/Premiere
  CPU; the snapshot normalizes it to `0..1` against the layer dimensions, so
  `ctx.get(Center) -> Point2` is host-agnostic. On the Premiere GPU path the
  host already delivers `0..1`.
- `precision = 0` renders the `Samples` slider as an integer even though the
  value is `f32`.

## 3. Typed enum popups

When a popup's options are also a Rust concept, declare them as a
`#[repr(u32)]` enum with `#[derive(prgpu::Popup)]` and point the popup at the
type. The read becomes typed instead of a raw `u32`, and out-of-range values
clamp to the first variant.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug, prgpu::Popup)]
#[repr(u32)]
pub enum Quality {
    #[option("Draft")]
    Draft = 0,
    #[option("Balanced")]
    Balanced = 1,
    #[option("High")]
    High = 2,
}

prgpu::params! {
    pub enum Params {
        #[popup(label = "Quality", options = Quality, default = Quality::High)]
        Qual,
    }
}
```

```rust
// in the pipeline:
match ctx.get(Qual) {
    Quality::Draft => 8,
    Quality::Balanced => 16,
    Quality::High => 32,
}
```

The derive generates `impl PopupOptions` (the `LABELS` shown in the host,
plus `from_index` / `to_index` with clamping) and `impl FromParamValue`, so
the snapshot coerces both AE 1-based integers and Premiere `Float32`
popups into the enum.

## 4. A routed multi-step wizard

Routes let you switch whole panels in and out based on a per-instance state.
Tag a group with `route = Name` and the generated `Router` enum gains a
variant; only the group(s) tagged with the active route are shown. An
`on_action` button handler receives an `ActionCtx` to navigate routes and
run background work.

This example shows a three-state wizard — *Idle → Processing → Done*:

```rust
prgpu::params! {
    pub enum Params {
        #[button(label = "Feedback", on_click = open_feedback)]
        Feedback,

        #[group("Settings", route = Idle, initial)]
        #[button(label = "Start", on_action = start_processing)]
        StartBtn,
        #[popup(label = "Output size", options = ["Small", "Medium", "Large"])]
        OutputSize,
        #[group(end)]

        #[group("Processing", route = Processing)]
        #[button(label = "Cancel", on_action = cancel_processing)]
        CancelBtn,
        #[label(text = format!("Working {}…", task_status()))]
        Progress,
        #[group(end)]

        #[group("Done", route = Done)]
        #[button(label = "Start again", on_action = reset)]
        RestartBtn,
        #[group(end)]

        #[checkbox(label = "Debug", default = false, debug_only)]
        Debug,
    }
}

fn open_feedback() {
    let _ = webbrowser::open("https://example.com/feedback");
}

fn start_processing(cx: &mut ActionCtx<Params>) {
    cx.spawn(crate::WorkTask::new(25), &["work"]);
    cx.goto(Router::Processing);
}

fn cancel_processing(cx: &mut ActionCtx<Params>) {
    cx.cancel_tag("work");
    cx.goto(Router::Idle);
}

fn reset(cx: &mut ActionCtx<Params>) {
    cx.cancel_tag("work");
    cx.goto(Router::Idle);
}

fn task_status() -> String {
    // Read from your BackgroundTask's status here.
    "0/25".to_string()
}
```

What this uses:

- **`route = Idle, initial`** — the `Settings` group is only visible while
  `Router::Idle` is active, and `Idle` is the default route on a fresh
  instance. The first declared route is the default if none is marked
  `initial`; `initial` is only valid together with `route = ...`.
- **`on_action = f`** — the handler takes `&mut ActionCtx<Params>` and can
  navigate (`cx.goto(Router::...)`), spawn background tasks
  (`cx.spawn(task, &["tag"])`), and cancel them (`cx.cancel_tag("tag")`).
  The simpler `on_click = f` form takes a plain `fn()` and is wrapped by the
  macro; use `on_action` whenever you need the context.
- **`text = format!(...)`** on a button — a *name-driven* button whose
  caption is re-evaluated on every UI refresh and pushed to the host via
  `PF_UpdateParamUI` (the native "frame x of y" pattern). The expression
  runs as `|_ctx| Into::<String>::into(expr)`, so it can call free functions
  like `task_status()` but cannot read `ctx`.
- **`#[label(text = ...)]`** — Drawbot-drawn static/dynamic text rendered in
  the ECW control area. Same expression form as the name-driven button, but
  painted by the adapter on `PF_Event_DRAW`. Omit `text` and bind the label
  dynamically against `ctx` in `Effect::ui` via `Ui::set_label` instead.
- **`Router`** — auto-generated as `pub enum Router { Idle, Processing,
  Done }`. A background task that finishes on a worker thread can call
  `Router::set(Router::Done)` directly; the main thread flushes the
  request to the per-instance route param on the next `UpdateParamsUi` and
  re-applies visibility. `Router::current().next()` / `.prev()` wrap around
  for simple next/prev navigation.

## 5. Wide-range sliders with a constrained handle

When a value's valid range is much larger than what is comfortable to drag,
give the slider a separate handle range. Here `Amplitude` is valid up to
`9999` but the draggable handle stays in `0..=100`; the user can still type
a larger value into the field.

```rust
#[slider(label = "Amplitude",
         range = 0.0..=9999.0,
         slider_range = 0.0..=100.0,
         default = 1.0, precision = 2)]
Amplitude,
#[slider(label = "Blur Length",
         range = 0.0..=9999.0,
         slider_range = 0.0..=10.0,
         default = 0.5, precision = 2)]
BlurLength,
```

`range` is the valid (clamp) range and must contain `default`.
`slider_range` is optional and defaults to `range`. `ctx.get(Amplitude)`
returns the full-range `f32`, not the handle range.

## 6. Dynamic visibility and live labels

Declarative groups fold the UI statically. For controls that appear or hide
based on *other* parameters or host capabilities, override `Effect::ui` and
add predicates against `Ctx`. This example hides a set of tint controls
when `TintMode` is `None` and gates an expansion control on hosts without
frame expansion:

```rust
fn ui(u: &mut Ui<Params>) {
    u.show_all(
        [Params::TintBlendMode, Params::TintStrength, Params::TintColor],
        |ctx| ctx.get(TintMode) != 0,
    );
    u.show(Params::AllowExpansion, |ctx| ctx.supports(Capability::FrameExpansion));
}
```

`u.show(Marker, |ctx| -> bool)` and `u.show_all([M; N], |ctx| -> bool)` are
re-evaluated on every `UpdateParamsUi` / `UserChangedParam`. The adapter
applies both the AE `INVISIBLE` flag and the AEGP dynamic-stream flag from
the same rule. See [`params_visibility.md`](../params_visibility.md) for
the full predicate and capability model.

For a label that depends on `ctx` (a name-driven button or `#[label]` whose
text needs live parameter values), use `Ui::set_label` instead of the
declarative `text = ...`:

```rust
fn ui(u: &mut Ui<Params>) {
    u.set_label(Params::StatusLabel, |ctx| format!("Strength: {:.0}", ctx.get(Strength)));
}
```

`#[label(text = ...)]` and `#[button(text = ...)]` cover the ctx-independent
case; `set_label` covers the ctx-dependent one.

## 7. A secondary image input

`#[layer]` adds a second image input (`PF_ADD_LAYER`). The marker carries a
`LAYER_INDEX: u32` constant; the adapter checks the layer out into the
matching invocation slot, and the pipeline asks whether it actually arrived
this frame via `ctx.layer_present`.

```rust
prgpu::params! {
    pub enum Params {
        #[layer(label = "Displacement Map", default = none)]
        DispMap,
        #[slider(label = "Amount", range = 0.0..=100.0, default = 25.0)]
        Amount,
    }
}
```

```rust
// in the pipeline:
let has_map = ctx.layer_present(DispMap::LAYER_INDEX);
```

- `default = none` (or omitted) leaves the input unassigned; `default =
  myself` maps to `PF_LayerDefault_MYSELF`.
- Layer params are **inert on Premiere** — the Premiere GPU host does not
  support layer parameters. Branch on `ctx.layer_present(...)` (which is
  `false` there) rather than on `is_premiere()`.
- A layer param has no readable snapshot value; presence is the only signal.

## 8. The escape hatch: `#[custom]`

When a parameter needs host SDK calls the DSL does not model, declare it
`#[custom(setup = path)]` and write the `Parameters::add_*` call yourself.
The macro calls your `setup` function with the `Parameters` handle and the
param id; everything else (snapshot, markers, dispatch) is still generated.

```rust
prgpu::params! {
    pub enum Params {
        #[custom(setup = add_arbitrary)]
        CustomThing,
        #[slider(label = "Strength", range = 0.0..=1.0, default = 0.5)]
        Strength,
    }
}

fn add_arbitrary(params: &mut ae::Parameters<Params>, id: Params) -> Result<(), ae::Error> {
    params.add_customized(id, "Custom",
        ae::ArbitraryDef::setup(|f| { let _ = f.set_default::<MyArb>(Default::default()); }),
        |p| { p.set_ui_flag(ae::ParamUIFlags::CONTROL, true); -1 })?;
    Ok(())
}
```

`#[custom]` and `#[label]` are the only kinds that do **not** require a
`label` key — `custom` because you supply the label inside your `setup`
function, `label` because its text *is* the label.

## See also

- [`01-introduction.md`](01-introduction.md) — orientation, setup, the
  minimal skeleton, rules of thumb.
- [`03-reference.md`](03-reference.md) — full attribute grammar, generated
  items, `Router` and `ActionCtx` APIs.
- [`effect_api.md`](../effect_api.md) — the `Effect` trait the generated
  `Params` plugs into.
- [`params_visibility.md`](../params_visibility.md) — `Effect::ui`
  predicates, capabilities, and `set_label`.
