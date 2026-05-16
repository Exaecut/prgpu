# PRGPU Automated Effect Testing

Offline GPU render harness for PRGPU effects. Write `cargo test` functions that
dispatch your real Slang kernel on Metal (macOS) or CUDA (Windows), download the
output, and diff it against a reference — without opening Premiere Pro or After Effects.

## Quick start

```bash
cargo test -p my-effect --test render_basic -- --nocapture
```

Your test creates a GPU context, uploads an image, dispatches your kernel,
downloads the result, and writes a PNG to `tests/output/`.

## Architecture

Your test file calls into `prgpu::testing`, which wraps `prgpu::kernels` (the
Slang kernels like `diff`) and `prgpu::gpu::backends` (Metal or CUDA). The flow
is:

1. **`GpuContext::create()`** — acquires the system GPU device and creates a command queue.
2. **`create_io_buffers()`** — allocates GPU input/output buffers through PRGPU's LRU cache.
3. **`upload_to_buffer()`** — copies host pixel data to the GPU via a staging buffer.
4. **`build_config()`** — assembles a `Configuration` with platform-specific handles.
5. **`your_kernel(&config, params)`** — dispatches the real Slang kernel on the GPU.
6. **`download_from_buffer()`** — reads back the output buffer to host memory.
7. **`write_png()`** / **`compute_metrics()`** / **`write_heatmap_png()`** — output and comparison.

## Module reference

| Module | What it provides |
|--------|-----------------|
| `prgpu::testing::context` | `GpuContext` — device creation, buffer allocation, host↔GPU transfer |
| `prgpu::testing::media` | Built-in test images: checkerboard, solid colour, horizontal gradient |
| `prgpu::testing::output` | `write_png()` — saves BGRA pixels as PNG (handles swizzle) |
| `prgpu::testing::scene` | `Scene`, `Layer`, `Transform`, `Timeline` — compositing model |
| `prgpu::testing::runner` | `RenderTest` — multi-frame render loop with PNG output |
| `prgpu::testing::compare` | `compute_metrics()`, `diff_heatmap_gpu()`, `write_heatmap_png()`, report writers |
| `prgpu::kernels::diff` | GPU diff kernel entry — blackbody heatmap with configurable smoothstep |
| `prgpu/shaders/diff.slang` | Slang diff kernel source |

## File layout per effect

```
effect-crate/
  tests/
    render_basic.rs   ← your test functions
    assets/           ← input images, reference images
    output/           ← generated PNGs, heatmaps, reports (gitignored)
    .gitignore        ← ignores /assets/ and /output/
```

Your effect's kernel is already public via `declare_kernel!`. Tests import it
directly — no extra wiring needed:

```rust
use my_effect::kernel::{my_kernel, MyParams};
```

## Built-in media

Three deterministic generators — no files required:

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
blue → orange → white. The transition is controlled by `smooth_a` and `smooth_b`,
which define where the smoothstep sigmoid ramp lives. Narrow the range to amplify
tiny differences; widen it for linear contrast.

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

### Metrics at a glance

All metrics are computed from raw pixel differences, independent of tolerance
(tolerance only affects the "pixels different" count).

| Metric | What it tells you |
|--------|------------------|
| `pixels_equal` | How many pixels are within tolerance |
| `pixels_different` | How many exceeded tolerance on any channel |
| `mean_absolute_error` | Average per-channel deviation (linear) |
| `max_absolute_error` | Worst single-channel error in the entire image |
| `root_mean_square_error` | Like MAE but penalises large errors more |
| `psnr` | RMSE in decibels — higher = closer to reference |

## Platform support

| Platform | Backend | GPU required |
|----------|---------|-------------|
| macOS (Apple Silicon) | Metal | Yes |
| Windows (NVIDIA GPU) | CUDA | Yes |
| Either without GPU | — | `GpuContext::create()` returns an error |

## Tutorials

- **[Tutorial 1 — Basic](tutorial-01-basic.md)**: Write your first GPU test, render a checkerboard, save a PNG.
- **[Tutorial 2 — Advanced](tutorial-02-advanced.md)**: Add a golden reference, generate heatmaps, load custom images, cross-tint testing.
- **[Tutorial 3 — Metrics Explained](tutorial-03-metrics.md)**: Understand tolerance, MAE, RMSE, PSNR, asymmetric thresholds, smoothstep sensitivity, and monotonicity — with real data from our test suite.

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
