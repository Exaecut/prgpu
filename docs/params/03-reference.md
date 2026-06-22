# `params!` — reference

The full grammar of `prgpu::params!`, the items it generates, and the
runtime APIs the generated code exposes. For orientation see
[`01-introduction.md`](01-introduction.md); for worked scenarios see
[`02-use-cases.md`](02-use-cases.md).

## Grammar

```rust
prgpu::params! {
    $vis enum $Name {
        $( $(#[$attr])*  $Variant, )*     // a kind attribute + variant name
    }
}
```

Each `$Variant` is one parameter. `$attr` is exactly one **kind attribute**
(see below) and optionally a preceding `#[group(...)]` opener or a trailing
`#[group(end)]` closer. A `#[group(...)]` opener applies to the *range* of
variants that follows it, up to the matching `#[group(end)]`; it is not a
kind attribute and a variant still needs its own kind.

`$vis` is propagated to the generated `enum $Name`, the marker structs, the
`Snapshot`, and the `Router` enum (when present).

## Kind attributes

Every variant carries exactly one kind attribute. `label = "..."` is
required for all kinds except `custom` and `label`.

### `#[slider(...)]`

A float slider. Reads as `f32`.

| Key            | Required | Form                  | Notes |
|----------------|----------|-----------------------|-------|
| `label`        | yes      | `"text"`              | |
| `range`        | yes      | `a..=b`               | valid (clamp) range; must contain `default` |
| `default`      | yes      | number                | compile error if outside `range` |
| `slider_range` | no       | `a..=b`               | handle drag range; defaults to `range` |
| `percent`      | no       | flag                  | host displays as a percentage; value unchanged |
| `precision`    | no       | integer (`i16`)      | decimal places shown; `0` renders integer |
| `debug_only`   | no       | flag                  | hidden in release builds |

```rust
#[slider(label = "Strength", range = 0.0..=100.0, default = 50.0, percent, precision = 3)]
Strength,
```

### `#[checkbox(...)]`

A boolean checkbox. Reads as `bool`.

| Key          | Required | Form      | Notes |
|--------------|----------|-----------|-------|
| `label`      | yes      | `"text"`  | |
| `default`    | no       | `bool`    | defaults to `false` |
| `supervise`  | no       | flag      | set `ParamFlag::SUPERVISE` (host re-renders on change) |
| `debug_only` | no       | flag      | hidden in release; the `#[checkbox(debug_only)]` param is wired as the effect's debug-view switch |

```rust
#[checkbox(label = "Invert", default = false)]
Invert,
#[checkbox(label = "Flicker", default = false, supervise)]
Flicker,
```

### `#[color(...)]`

A colour swatch. Reads as `Color` (RGBA, channels normalized `0..1`).

| Key       | Required | Form                | Notes |
|-----------|----------|---------------------|-------|
| `label`   | yes      | `"text"`            | |
| `default` | yes      | `"#RRGGBB"` or `"#RRGGBBAA"` | alpha defaults to `255` (fully opaque) |

```rust
#[color(label = "Tint Color", default = "#FF8040")]
Tint,
```

### `#[angle(...)]`

An angle dial. Reads as `f32` (host angle units; `prgpu::DEG_TO_RAD` is
exported for radian conversion).

| Key       | Required | Form   | Notes |
|-----------|----------|--------|-------|
| `label`   | yes      | `"text"` | |
| `default` | no       | number | defaults to `0.0` |

```rust
#[angle(label = "Gradient Angle", default = 0.0)]
GradientAngle,
```

### `#[point(...)]`

A 2D point. Reads as `Point2` (`x`, `y` normalized to `0..1` against the
layer dimensions on the CPU path; already `0..1` on the Premiere GPU path).

| Key       | Required | Form        | Notes |
|-----------|----------|-------------|-------|
| `label`   | yes      | `"text"`    | |
| `default` | yes      | `(x, y)`    | two floats |

```rust
#[point(label = "Center", default = (50.0, 50.0))]
Center,
```

### `#[popup(...)]`

A dropdown. Reads as `u32` for inline options, or the named enum for enum
options. **Indices are 0-based on every host** (the macro subtracts 1 from
AE's 1-based SDK value and coerces Premiere's `Float32` popups).

| Key       | Required | Form                              | Notes |
|-----------|----------|-----------------------------------|-------|
| `label`   | yes      | `"text"`                          | |
| `options` | yes      | `["A", "B", ...]` or `EnumType`   | inline array (reads `u32`) or a `#[derive(prgpu::Popup)]` enum (reads that enum) |
| `default` | yes      | integer (inline) or `Enum::Var`   | |
| `debug_only` | no    | flag                              | hidden in release |

```rust
#[popup(label = "Mode", options = ["A", "B", "C"], default = 1)]
Mode,        // ctx.get(Mode) -> u32  (0-based)

#[popup(label = "Quality", options = Quality, default = Quality::High)]
Qual,        // ctx.get(Qual) -> Quality
```

### `#[blend_mode(...)]`

Sugar for `#[popup(options = prgpu::BlendMode, default = BlendMode::Variant)]`.
Reads as `BlendMode`. The `default` is a bare variant name (`Add`,
`Multiply`, `Screen`, ...); see `prgpu::BlendMode` for the full list.

| Key       | Required | Form        | Notes |
|-----------|----------|-------------|-------|
| `label`   | yes      | `"text"`    | |
| `default` | yes      | `Variant`   | a `BlendMode` variant |

```rust
#[blend_mode(label = "Blend Mode", default = Add)]
BlendMode,
```

### `#[button(...)]`

A push button. Reads as `()` (no snapshot value). Exactly one of
`on_click` / `on_action` is conventional; `on_action` wins if both are set.

| Key         | Required | Form                  | Notes |
|-------------|----------|-----------------------|-------|
| `label`     | yes      | `"text"`              | caption when no `text` |
| `on_click`  | no       | `path` to `fn()`      | legacy no-arg handler; wrapped to ignore the context |
| `on_action` | no       | `path` to `fn(&mut ActionCtx<$Name>)` | context-aware handler: route navigation + background tasks |
| `text`      | no       | Rust expr             | live caption, re-evaluated each UI refresh; `|_ctx| Into::<String>::into(expr)` |

```rust
#[button(label = "Feedback", on_click = open_feedback)]
Feedback,
#[button(label = "Start", on_action = start_processing)]
StartBtn,
#[button(label = "Status", text = format!("Remaining: {}", remaining_count()))]
StatusLabel,
```

A `text = ...` button is *name-driven*: the adapter pushes the evaluated
caption into the param name and calls `PF_UpdateParamUI` (the native
"frame x of y" pattern). Without `text`, the static `label` is the caption.

### `#[layer(...)]`

A secondary image input (`PF_ADD_LAYER`). Reads as `()` — presence is the
only signal, resolved at checkout via `ctx.layer_present(Marker::LAYER_INDEX)`.
Inert on Premiere (layer params are unsupported there).

| Key       | Required | Form            | Notes |
|-----------|----------|-----------------|-------|
| `label`   | yes      | `"text"`        | |
| `default` | no       | `myself` / `none` | `myself` → `PF_LayerDefault_MYSELF`; `none` or omitted → unassigned |

```rust
#[layer(label = "Displacement Map", default = none)]
DispMap,
// ctx.layer_present(DispMap::LAYER_INDEX) -> bool
```

The marker gets an inherent `pub const LAYER_INDEX: u32`, assigned in
declaration order across all `#[layer]` params (0, 1, 2, ...).

### `#[custom(...)]`

The escape hatch for host SDK calls the DSL does not model. No `label`
key — you supply the label inside your `setup` function. Reads as `()`.

| Key     | Required | Form        | Notes |
|---------|----------|-------------|-------|
| `setup` | yes      | `path` to `fn(&mut Parameters<$Name>, $Name) -> Result<(), ae::Error>` | called by the generated `register` with the params handle and the variant id |

```rust
#[custom(setup = add_arbitrary)]
CustomThing,
```

### `#[label(...)]`

Drawbot-drawn static/dynamic text in the ECW control area. No `label`
key — the `text` *is* the label. Reads as `()`.

| Key   | Required | Form     | Notes |
|-------|----------|----------|-------|
| `text`| no       | Rust expr | `|_ctx| Into::<String>::into(expr)`, evaluated each refresh; omit and bind via `Ui::set_label` in `Effect::ui` for ctx-dependent text |

```rust
#[label(text = format!("Working {}…", task_status()))]
Progress,
```

## Group attribute

`#[group(...)]` is **not** a kind attribute; it frames a range of variants.
A group opener is followed by one or more variants (each with their own
kind), then a matching `#[group(end)]`. Groups nest.

### Opener

`#[group("Title", collapsed?, route = Name?, initial?)]`

| Key         | Required | Form      | Notes |
|-------------|----------|-----------|-------|
| (positional)| yes      | `"Title"` | group header text |
| `collapsed` | no       | flag or `= bool` | default expanded (`false`) |
| `route`     | no       | `= Ident` | tie the group to a route; only visible while that route is active |
| `initial`   | no       | flag      | only valid with `route = ...`; marks this route as the default on a fresh instance |

```rust
#[group("Settings", route = Idle, initial)]
...
#[group(end)]
```

### Closer

`#[group(end)]` — required for every opener; a compile error is raised on
an unclosed group.

### Routes

The distinct route names from all `route = X` openers (in declaration
order) become the variants of the generated `Router` enum. The first route
is the default unless one is marked `initial`. The active route is stored
**per instance** in an auto-injected hidden popup param (project-persisted).

## Universal keys

| Key         | Applies to                | Effect |
|-------------|---------------------------|--------|
| `label`     | all kinds except `custom`, `label` | the host caption |
| `debug_only`| any kind                  | hidden (collapsed + `INVISIBLE`) in release builds; for `checkbox`, also designates the effect's debug-view switch |

## `#[derive(prgpu::Popup)]`

Turns a `#[repr(u32)]` unit enum into popup options. Each variant takes
`#[option("Label")]` (the label defaults to the variant name if omitted).

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug, prgpu::Popup)]
#[repr(u32)]
pub enum Quality {
    #[option("Draft")]    Draft = 0,
    #[option("Balanced")] Balanced = 1,
    #[option("High")]      High = 2,
}
```

Generates `impl PopupOptions` (`LABELS`, `from_index` with out-of-range
clamping to the first variant, `to_index`) and `impl FromParamValue`
(coerces `Index` and Premiere `Float32` snapshots into the enum).

## Generated items

For a `pub enum Params`, the macro emits:

| Item | Description |
|------|-------------|
| `enum Params` | `#[repr(usize)]`, `#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]`; discriminants start at `1` (`0` is the AE input layer) |
| `struct $Variant;` | one zero-sized marker per param; `impl prgpu::Param` with `type Spec = Params`, `type Value`, `const ID: Params` |
| `impl $LayerVariant { pub const LAYER_INDEX: u32 }` | only on `#[layer]` markers, in declaration order |
| `__Snapshot` | `#[doc(hidden)]`, `Copy`, indexed by `Params`; `impl Snapshot<Params>` (`value`, `set`) |
| `enum Router` + `impl Router` + `impl Route` | only when at least one `route = ...` is declared |
| `impl ParamsSpec for Params` | registration + snapshot + dispatch (see below) |
| `impl SetupParams for Params` | legacy bridge so the old `Effect` path keeps compiling |

### `ParamsSpec` surface

The constants and methods the adapters drive:

| Member | What it is |
|--------|------------|
| `const COUNT: usize` | total slots including group markers and the hidden route store |
| `const ALL: &'static [Self]` | leaf params only (no group markers, no route store); used for bulk show/hide |
| `const DEBUG_PARAM: Option<Self>` | the `#[checkbox(debug_only)]` param, if any |
| `const LAYER_PARAMS: &'static [Self]` | `#[layer]` params in declaration order |
| `const LABEL_PARAMS: &'static [Self]` | `#[label]` params in declaration order |
| `const NAME_DRIVEN_PARAMS: &'static [Self]` | `#[button(text = ...)]` params |
| `const ROUTED_GROUPS: &'static [(Self, u32)]` | `(group-start marker, route index)` per `#[group(route = ...)]` |
| `const ROUTE_PARAM: Option<Self>` | the hidden route store param, if routes are declared |
| `type Snapshot` | the `__Snapshot` storage |
| `fn register(&mut Parameters<Self>)` | host registration, in declaration order |
| `fn snapshot_cpu(...)` / `fn snapshot_gpu(...)` | per-frame snapshot (host quirks resolved here) |
| `fn buttons()` | `&[(Self, fn(&mut ActionCtx<Self>))]` dispatch table |
| `fn contribute_labels(&mut Ui<Self>)` | pushes `#[label(text=...)]` and `#[button(text=...)]` bindings |

## Reading values

The read side is `Ctx::get`, fully typed through the marker:

```rust
ctx.get(Strength)   // -> f32
ctx.get(Invert)      // -> bool
ctx.get(Tint)        // -> Color
ctx.get(Anchor)      // -> Point2
ctx.get(Mode)        // -> u32      (inline popup)
ctx.get(Qual)        // -> Quality  (enum popup)
ctx.get(BlendMode)   // -> BlendMode
```

Unset snapshot slots and out-of-variant coercions fall back to `Default`
(`0.0`, `false`, `Color::default()`, the enum's first variant). For layer
params, use `ctx.layer_present(Marker::LAYER_INDEX) -> bool`.

## Button handlers: `ActionCtx`

`#[button(on_action = f)]` hands the handler an `&mut ActionCtx<Params>`:

```rust
pub struct ActionCtx<P: ParamsSpec> { /* ... */ }

impl<P: ParamsSpec> ActionCtx<P> {
    pub fn goto<R: Route>(&mut self, route: R);
    pub fn spawn<T: BackgroundTask>(&mut self, task: T, tags: &[&'static str]) -> TaskHandle;
    pub fn cancel(&mut self, id: TaskId);
    pub fn cancel_tag(&mut self, tag: &'static str);
}
```

`goto` requests a route change; the adapter flushes it to the per-instance
route param and re-applies visibility after the handler returns. `spawn`
runs a cooperative, cancellable `BackgroundTask` (poll-based) tagged for
later lookup/cancellation. `#[button(on_click = f)]` (a plain `fn()`) is
wrapped to ignore the context — use it only for side effects that need no
navigation or task control.

## The `Router` API

Generated only when at least one `route = ...` is declared:

```rust
pub enum Router { /* one variant per distinct route, in declaration order */ }

impl Router {
    pub const COUNT: usize;
    pub fn current() -> Router;          // active route for this instance
    pub fn set(route: Router);           // request a change (any thread)
    pub fn name(self) -> &'static str;
    pub fn index(self) -> u32;
    pub fn next(self) -> Router;         // wraps around
    pub fn prev(self) -> Router;         // wraps around
}
```

`Router::current()` reads the per-call active route (seeded by the adapter
from the hidden route param at the start of each command). `Router::set`
records a request that the main thread flushes on the next
`UpdateParamsUi`; a background-task worker thread can call it to navigate
(e.g. task done → `Router::Done`). `Route::INITIAL` is the route marked
`initial`, else the first declared route.

## Constraints and compile errors

The macro validates the declaration at compile time. These all raise a
compile error (not a runtime panic):

| Mistake | Error |
|---------|-------|
| Variant with no kind attribute | `` `X` is missing a parameter-kind attribute `` |
| Two kind attributes on one variant | `more than one parameter-kind attribute on a variant` |
| Unknown attribute | `` expected one of slider/checkbox/color/angle/point/popup/blend_mode/button/layer/custom/label/group `` |
| Missing `label` on a kind that needs it | `` `X` is missing `label = "..."` `` |
| Slider `default` outside `range` | `default {d} is outside range {a}..={b}` |
| Slider missing `range` / `default` | `slider needs range = a..=b` / `slider needs default = ..` |
| Colour not `#RRGGBB` / `#RRGGBBAA` | `colour must be #RRGGBB or #RRGGBBAA` |
| Point not a 2-tuple | `expected (x, y)` |
| Popup `options` neither array nor path | `popup options must be a string array or an enum path` |
| Popup missing `default` | `popup needs default = ..` |
| `blend_mode` missing `default` | `blend_mode needs default = Variant` |
| `layer` default not `myself`/`none` | `layer default must be myself or none` |
| `custom` missing `setup` | `custom needs setup = path` |
| Duplicate variant name | `duplicate parameter X` |
| `#[group(end)]` with no open group | `#[group(end)] without a matching #[group("...")]` |
| Unclosed group | `unclosed #[group("...")] — add #[group(end)]` |
| `initial` without `route` | `initial is only valid together with route = Name` |
| No parameters at all | `params! needs at least one parameter` |
| Trailing kind attribute with no variant | `trailing parameter-kind attribute with no variant name` |

A kind attribute belongs to the **next** variant identifier. A floating kind
attribute (one with no variant name before a `#[group(end)]` or another
attribute) is parsed as an attribute of the following variant and triggers
"more than one parameter-kind attribute on a variant" — always name the
variant immediately after its kind attribute.

## Cross-references

- `prgpu::Popup` derive — source of `PopupOptions` / `FromParamValue` for
  enum popups (`prgpu-macro/src/popup.rs`).
- `prgpu::BlendMode` — the built-in popup enum used by `#[blend_mode]`
  (`prgpu/src/params/blend.rs`).
- `prgpu::params::ParamsSpec` — the trait the generated enum implements
  (`prgpu/src/params/traits.rs`).
- `prgpu::effect::{Ctx, ActionCtx, Route, Ui}` — the read, action, route,
  and visibility surfaces (`prgpu/src/effect/`).
- [`01-introduction.md`](01-introduction.md),
  [`02-use-cases.md`](02-use-cases.md),
  [`effect_api.md`](../effect_api.md),
  [`params_visibility.md`](../params_visibility.md).
