# Migrating an existing effect to the high-level API

Mindglow shrank from 865 lines (`lib.rs` + `gpu.rs`) to 270 lines through
this migration. The mapping is mechanical — most existing logic moves
into trait methods without behavioural change.

## Old → new

| Old (per-effect handwritten)                   | New (declarative)                         |
|------------------------------------------------|--------------------------------------------|
| `impl AdobePluginGlobal for Plugin { ... }`    | `Effect::params` + `Effect::ui` + adapter trampoline |
| `impl pr::GpuFilter for PremiereGPU { ... }`   | `pub type PremiereGPU = adobe::premiere::GpuFilterAdapter<E>;` |
| Manual `Cmd_GlobalSetup` (logger / pixel formats / AEGP register) | adapter handles via `EffectDescriptor` |
| Manual `Cmd_About` strings                     | `EffectDescriptor::new("X").about("...").version(...)` |
| Manual `Cmd_UpdateParamsUi` AE PF flag flips + AEGP `set_dynamic_stream_flag` | `Effect::ui` + `ParamApi::visibility` |
| Manual `Cmd_UserChangedParam` if/else by `param_index` | `ParamApi::actions::on_click(P, callback)` |
| Manual `SmartPreRender` expansion arithmetic   | `Effect::expansion` returning `ExpansionExtent` |
| Manual frame-data extractors (`from_cpu` / `from_gpu` per host) | `Effect::frame_data(ctx)` + `Params::from_context(&ctx)` |
| Manual `Configuration::cpu(...)` / `Configuration::effect(...)` mutation per pass | `RenderGraph::add_pass` / `add_mip_chain` + `ConfigBuilder` (internal) |
| Manual `prgpu::cpu::buffer::get_or_create_with_mips` / `prgpu::gpu::buffer::get_or_create_with_mips` | `RenderGraph::declare_mip_pyramid` |
| Manual `prepare_source_copy` for Premiere alias | `RenderGraph::set_source_policy(SourcePolicy::SnapshotIfAliased { tag })` |
| Manual `bloom_prefilter(&cfg, params)` / `bloom_prefilter_cpu(...)` per backend | `Kernel<P>` descriptor + graph executor picks backend |
| `is_premiere()` / `is_after_effects()` checks | `host.supports(Capability::*)` |
| Manual licence init in `Cmd_GlobalSetup` + checks before each render | `LicenseGate::initialize` / `is_valid` |

## Mechanical migration steps

### 1. Define `FrameData`

Pull the per-pass kernel-params + any host-derived state (frame index,
time, ext_x/ext_y) into one `Copy + Send + Sync + 'static` struct.

```rust
#[derive(Clone, Copy)]
pub struct FrameData {
    pub prefilter: BloomPrefilterParams,
    pub upsample: BloomUpsampleParams,
    pub composite: MindglowCompositeParams,
    pub quality: u32,
    pub frame_index: u32,
    pub ext_x: i32,
    pub ext_y: i32,
}
```

### 2. Implement `LicenseGate` (or use `NoLicenseGate`)

If your effect has no licence check:

```rust
impl Effect for MyEffect { type License = prgpu::effect::NoLicenseGate; ... }
```

Otherwise wrap your existing licence calls. See [`license_gate.md`](license_gate.md).

### 3. Implement `Effect`

```rust
#[derive(Default)]
pub struct MyEffect;

impl Effect for MyEffect {
    type Params = Params;
    type FrameData = FrameData;
    type License = MyLicense;

    fn descriptor() -> EffectDescriptor { /* metadata */ }
    fn params(p, in_data, out_data) -> Result<(), ae::Error> { Params::setup(p, in_data, out_data) }
    fn ui(api: &mut ParamApi<Params>) -> Result<(), ae::Error> { /* visibility + actions */ Ok(()) }
    fn frame_data(ctx: FrameDataContext<Params>) -> Result<FrameData, ae::Error> { /* ... */ }
    fn expansion(ctx: ExpansionContext<Params>) -> Result<ExpansionExtent, ae::Error> { /* ... */ }
    fn pipeline(g: &mut RenderGraph<FrameData>) { /* ... */ }
}
```

### 4. Add the adapter trampoline + macros

```rust
#[derive(Default)]
struct Plugin(prgpu::adobe::ae::EffectAdapter<MyEffect>);

impl AdobePluginGlobal for Plugin {
    fn params_setup(&self, p, i, o) -> Result<(), ae::Error> { self.0.params_setup(p, i, o) }
    fn handle_command(&mut self, c, i, o, p) -> Result<(), ae::Error> { self.0.handle_command(c, i, o, p) }
}

ae::define_effect!(Plugin, (), Params);

pub type PremiereGPU = prgpu::adobe::premiere::GpuFilterAdapter<MyEffect>;
premiere::define_gpu_filter!(PremiereGPU);
```

### 5. Delete `gpu.rs`

The Premiere GPU adapter handles the entire `pr::GpuFilter::render` body
through the graph + `Effect::frame_data`. The handwritten `gpu.rs` file
becomes redundant.

### 6. Remove the manual `handle_command` match

Every selector arm collapses into the trait methods above.

### 7. Update tests

Tests that referenced `mod gpu::PremiereGPU` now reference
`crate::PremiereGPU` (the type alias). Tests that used `PremiereGPU` as a
value (the unit struct) must use `PremiereGPU::default()`.

## Backward compatibility for kernels

`declare_kernel!(name, P)` still emits the legacy top-level functions
(`name(cfg, params)`, `name_cpu(...)`, `NAME_CPU_DISPATCH_TILE`) as
`#[deprecated]` wrappers. Effects that haven't migrated yet keep
compiling with deprecation warnings instead of errors.

`kernel_params!` is unchanged from a caller's perspective. The macro
internally now layers `#[gpu_struct]` and auto-implements `KernelParams`,
but `from_cpu` / `from_gpu` keep the same signatures. Migration adds
`from_context` as a host-agnostic shorthand for use inside
`Effect::frame_data`.

## Verifying parity

Run any GPU render tests against a baseline output PNG before and after
migration. The reference output should be byte-identical — every behavior
the legacy code expressed (popup normalisation, BGRA layout, Premiere
alias snapshot, mip chain dispatch order) lives in prgpu now and produces
the same kernel inputs.

For mindglow, the regression bookmark is
`.kilo/plans/mindglow_baseline.png` (SHA256 `AC59...2716`); the
`mindglow/tests/render_basic.rs` test rerun matches the baseline exactly
post-migration.
