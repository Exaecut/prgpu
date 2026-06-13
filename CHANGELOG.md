# Changelog

## 0.2.0 — 2026-06-13

### Breaking: declarative Effect API (v2)

The entire effect-authoring surface has been redesigned around three declarative
macros and a simplified trait. See `docs/prgpu-audit/` for the full migration
guide.

- **`params!`** — single-source-of-truth parameter declaration with typed
  markers, per-frame `Snapshot`, `ParamsSpec`, and `Ctx::get(Marker)`.
- **`kernel!`** — one macro replaces `kernel_params!` + `declare_kernel!`;
  auto-padded GPU structs, `FromCtx` extraction, build-time Slang ABI check.
- **`register_effect!`** — one-line effect registration generating the AE
  `define_effect!` + Premiere `define_gpu_filter!` boilerplate.
- **`Effect` v2** — no more `FrameData`, `License`, `frame_data()`, or
  `params()`. Everything flows through `Ctx<P>` and `Graph<P>`.
- **`Graph<P>`** — pass-builder API with method chaining: `g.pass(kernel)`,
  `.reads()`, `.writes()`, `.params()`, `.when()`. Mip chains via
  `g.mip_chain()`.
- **`Ui<P>`** — snapshot-driven visibility rules replacing `ParamApi` /
  `VisibilityBuilder` / `ActionBuilder`.
- **prelude** — clean single-import surface: `use prgpu::prelude::*;`.

### Removed APIs

- `kernel_params!`, `declare_kernel!`, `include_shader!` (replaced by `kernel!`)
- `RenderGraph<F>`, `PassContext<F>`, `MipPyramidCtx<F>` (replaced by `Graph<P>`)
- `ParamApi`, `VisibilityBuilder`, `ActionBuilder`, `ActionContext` (replaced by `Ui<P>`)
- `FrameDataContext`, `ExpansionContext` (replaced by `Ctx<P>`)
- `SetupParams`, `CpuParams`, `FromParam`, `get_param`, `register_gpu_param_indices` (replaced by `ParamsSpec`)
- `GpuDispatchFn`, `CpuRenderFn` type aliases (folded into `Kernel`)
- `Effect::FrameData`, `Effect::License` associated types
- `Effect::frame_data`, `Effect::params` methods
- Manual `_pad` fields on GPU structs (auto-injected by `gpu_struct`)
- Per-effect `gpu_backend` / `with_premiere` cfg emissions

### Other changes

- Per-frame `log::info!` → `log::debug!` (logging diet)
- Module documentation pass
- `NoLicenseGate` → `NoLicense`
- `prgpu-build`: removed transitional cfg emissions
- Manual `Configuration` assembly replaced by `ConfigBuilder`
