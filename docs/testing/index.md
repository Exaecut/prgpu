# PRGPU Automated Effect Testing

Offline GPU render harness for PRGPU effects. Write `cargo test` functions that
dispatch your real Slang kernel on Metal (macOS) or CUDA (Windows), download the
output, and diff it against a reference — without opening Premiere Pro or After Effects.

## Quick start

```rust
use prgpu::testing::{HostBuilder, ParamValue, builtin_checkerboard, write_png};
use my_effect::gpu::PremiereGPU;
use my_effect::params::Params;

#[test]
fn render_basic() {
    let (w, h) = (512, 512);
    let input = builtin_checkerboard(w, h);

    let ctx = HostBuilder::<PremiereGPU, Params>::new(PremiereGPU, input, w, h)
        .param(Params::Strength, ParamValue::float(100.0))
        .param(Params::Tint, ParamValue::color(0, 0, 255, 255))
        .param(Params::ExpandFrame, ParamValue::bool(false))
        .build()
        .expect("HostContext");

    let output = ctx.start().expect("render chain");
    write_png("tests/output/basic.png", &output, w, h, 4).expect("write");
}
```

```bash
cargo test -p my-effect --test render_basic -- --nocapture
```

The `HostContext` runs the **full Premiere render chain** on the real GPU:
`global_init` → `get_frame_dependencies` → `precompute` → `render` → `global_destroy`.
Your Slang kernel receives the same `Configuration`, `FrameParams`, and user
parameters it would get inside Premiere Pro.

## Architecture

Your test calls `HostBuilder` which constructs a `HostContext`. The context
owns a real GPU device, mock Premiere FFI objects (`GpuFilterData`, `RenderParams`,
`PPixHand`, suites), and your effect's `PremiereGPU` instance. Calling `.start()`
executes the full render chain end-to-end.

For quick smoke tests that bypass the Premiere FFI (no `GpuFilter` trait, no
mock suites), the lower-level `GpuContext` + direct kernel dispatch path is
also available. Both paths run the same Slang kernel on the same GPU hardware.

## Module reference

| Module | What it provides |
|--------|-----------------|
| `prgpu::testing::host` | `HostContext`, `HostBuilder`, `ParamValue`, `pixel_format` — full Premiere path |
| `prgpu::testing::context` | `GpuContext` — direct GPU device creation, buffer allocation, transfer |
| `prgpu::testing::media` | Built-in images: `builtin_checkerboard()`, `builtin_solid_color()`, `builtin_gradient_h()` |
| `prgpu::testing::output` | `write_png()` — BGRA→RGBA swizzle + PNG save |
| `prgpu::testing::scene` | `Scene`, `Layer`, `Transform`, `Timeline` — compositing model |
| `prgpu::testing::runner` | `RenderTest` — multi-frame render loop with PNG output |
| `prgpu::testing::compare` | `compute_metrics()`, `diff_heatmap_gpu()`, `write_heatmap_png()`, JSON/txt reports |
| `prgpu::kernels::diff` | Built-in GPU diff kernel — blackbody heatmap with configurable smoothstep |

## Two rendering paths

| Path | Entry point | Premiere FFI | When to use |
|------|------------|-------------|-------------|
| **HostBuilder** | `HostContext::start()` | Full mock: `GpuFilterData`, `RenderParams`, `PPixHand`, suites | Default. Matches what Premiere does. |
| **Direct** | `GpuContext` + kernel dispatch | None — builds `Configuration` manually | Quick smoke tests, prototyping |

The HostBuilder path exercises `PremiereGPU::render()` exactly as Premiere
calls it. Parameters flow through `PlaygroundParams::from_gpu()` which reads
from the mock `VideoSegmentSuite`. The kernel receives the same `Configuration`
produced by `Configuration::effect()`.

## Effect-side requirements

For the **HostBuilder** path, your effect must expose its `GpuFilter` type:

```rust
// src/lib.rs
pub mod kernel;
pub mod gpu;       // was `mod gpu`
pub mod params;    // was `mod params`

// src/gpu.rs
#[derive(Default)]
pub struct PremiereGPU;   // was `struct PremiereGPU`
```

The `declare_kernel!` macro in `kernel.rs` already makes the kernel dispatch
function public — no changes needed there.

For the **direct** path, only `pub mod kernel` is needed (the kernel function
is already public via `declare_kernel!`).

## File layout per effect

```
effect-crate/
  tests/
    render_basic.rs   ← your test functions
    assets/           ← input images, reference images
    output/           ← generated PNGs, heatmaps, reports (gitignored)
    .gitignore        ← ignores /assets/ and /output/
```

## Built-in media

```rust
let data = builtin_checkerboard(256, 256);          // 32 px tiles
let data = builtin_solid_color(256, 256, Rgba8::RED);
let data = builtin_gradient_h(256, 256, Rgba8::BLACK, Rgba8::WHITE);
```

## Comparison engine

Two paths, same `DiffConfig`:

| Mode | Function | Use for |
|------|----------|---------|
| CPU metrics | `compute_metrics()` | Numeric report: MAE, RMSE, PSNR, pixel counts |
| GPU heatmap | `diff_heatmap_gpu()` | Visual diff on GPU (Slang kernel) |
| CPU heatmap | `write_heatmap_png()` | Same visual output, no GPU dispatch |

Both heatmap paths produce a **blackbody colormap**: black → dark blue → bright
blue → orange → white. Controlled by `smooth_a` and `smooth_b` smoothstep bounds.

```rust
let config = DiffConfig {
    tolerance_r: 0.01,    // per-channel pass/fail threshold
    tolerance_g: 0.01,
    tolerance_b: 0.01,
    tolerance_a: 0.01,
    smooth_a: 0.0,         // heatmap: zero error → black
    smooth_b: 1.0,         // heatmap: max error → white
};
```

## Platform support

| Platform | Backend | GPU required |
|----------|---------|-------------|
| macOS (Apple Silicon) | Metal | Yes |
| Windows (NVIDIA GPU) | CUDA | Yes |
| Either without GPU | — | `GpuContext::create()` returns an error |

## Tutorials

- **[Tutorial 1 — Basic](tutorial-01-basic.md)**: Write your first GPU test with HostBuilder, render a checkerboard, save a PNG.
- **[Tutorial 2 — Advanced](tutorial-02-advanced.md)**: Add a golden reference, generate heatmaps, load custom images, cross-tint testing.
- **[Tutorial 3 — Metrics Explained](tutorial-03-metrics.md)**: Understand tolerance, MAE, RMSE, PSNR, asymmetric thresholds, smoothstep sensitivity, and monotonicity.

## Debug checklist

| Symptom | Likely cause |
|---------|-------------|
| `GpuContext::create()` error | No GPU or driver mismatch |
| `STATUS_ACCESS_VIOLATION` on CUDA | `nvcuda.dll` absent or wrong CUDA version |
| Kernel returns error | Buffer sizes mismatched, null pointers, or PTX incompatible with GPU |
| Output is all black | Shader compiled but didn't write to `dst` — check binding indices |
| `compute_metrics()` size mismatch | Input and reference have different dimensions or bit depth |
| Heatmap is uniform colour | Image has uniform error. Test on a photo to see real gradients. |
| GPU heatmap disagrees with CPU metrics | Float rounding in `TextureView.Load()` vs `/ 255.0` |
| Params all zero despite `.param()` calls | `GpuFilterData::param()` subtracts 1 internally; `HostBuilder` adjusts automatically |
