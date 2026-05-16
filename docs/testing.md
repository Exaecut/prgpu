# PRGPU Automated Effect Testing

Offline GPU render harness for PRGPU effects. Write `cargo test` functions that
dispatch your real Slang kernel on Metal (macOS) or CUDA (Windows), download the
output, and diff it against a reference — all without opening Premiere Pro or
After Effects.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  playground/tests/render_basic.rs         ← your test file  │
│  ┌──────────────────────────────────────┐                   │
│  │ fn render_checkerboard_tint() {     │                   │
│  │   let gpu = GpuContext::create();   │                   │
│  │   let input = builtin_checkerboard(…);                   │
│  │   let (in_buf, out_buf) = gpu.create_io_buffers(…);     │
│  │   gpu.upload_to_buffer(&in_buf, &input, …);              │
│  │   let config = gpu.build_config(&in_buf, &out_buf, …);  │
│  │   playground(&config, params); // ← real GPU dispatch   │
│  │   let output = gpu.download_from_buffer(&out_buf, …);   │
│  │   write_png("output.png", &output, …);                   │
│  │   let report = compute_metrics(&output, &ref, …);       │
│  │   write_heatmap_png("heatmap.png", …);                   │
│  │   write_report_json("report.json", &report);            │
│  │ }                                                        │
│  └──────────────────────────────────────┘                   │
│                      │                                      │
│  ┌───────────────────▼──────────────────────┐               │
│  │       prgpu::testing                     │               │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐  │               │
│  │  │ context │ │ compare  │ │  output  │  │               │
│  │  │ GpuCtx  │ │ metrics  │ │ PNG/json │  │               │
│  │  │ buffers │ │ heatmap  │ │ reports  │  │               │
│  │  └────┬────┘ └────┬─────┘ └──────────┘  │               │
│  └───────┼───────────┼─────────────────────┘               │
│          │           │                                      │
│  ┌───────▼───────────▼─────────────────────┐               │
│  │         prgpu::kernels                  │               │
│  │  ┌────────┐  ┌──────────────┐           │               │
│  │  │  mip   │  │     diff     │           │               │
│  │  │downsample│ │  (heatmap)  │           │               │
│  │  └────────┘  └──────┬───────┘           │               │
│  └──────────────────────┼──────────────────┘               │
│                         │                                   │
│  ┌──────────────────────▼──────────────────┐               │
│  │     prgpu::gpu::backends                │               │
│  │   Metal (macOS) / CUDA (Windows)        │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Module overview

| Module | Role |
|--------|------|
| `prgpu::testing::context` | `GpuContext` — GPU device creation, buffer allocation, host↔GPU transfer |
| `prgpu::testing::media` | Built-in test images: checkerboard, solid colour, horizontal gradient |
| `prgpu::testing::output` | `write_png()` — saves tightly-packed BGRA to PNG (handles BGRA→RGBA swizzle) |
| `prgpu::testing::scene` | `Scene`, `Layer`, `Transform`, `Timeline` — compositing model for future use |
| `prgpu::testing::runner` | `RenderTest` — multi-frame render loop with PNG output |
| `prgpu::testing::compare` | `compute_metrics()`, `diff_heatmap_gpu()`, `write_heatmap_png()`, `write_report_json()`, `write_report_txt()` |
| `prgpu::kernels::diff` | `declare_kernel!(diff, DiffParams)` — GPU diff kernel entry point |
| `prgpu/shaders/diff.slang` | Blackbody heatmap Slang kernel with configurable smoothstep |

### File layout per effect

```
effect-crate/
  src/
    lib.rs          ← ae::define_effect! + pr::define_gpu_filter!
    gpu.rs          ← PremiereGPU impl
    kernel.rs       ← kernel_params! + declare_kernel!
    params.rs       ← parameter definitions
  shaders/
    effect.slang    ← your compute kernel
  tests/
    render_basic.rs ← your test functions
    assets/         ← input images (PNG/JPEG), reference images
    output/         ← generated PNGs, heatmaps, reports
    .gitignore      ← ignores /assets/ and /output/
```

### Effect-side setup

Your effect must expose its kernel publicly. The `declare_kernel!` macro in `kernel.rs`
already generates a `pub fn` for the GPU dispatch:

```rust
// In kernel.rs — already public via declare_kernel!
pub unsafe fn playground(config: &Configuration, user_params: PlaygroundParams) -> Result<(), &'static str>;
```

Your tests import it directly:

```rust
use playground::kernel::{playground, PlaygroundParams};
```

No extra code required on the effect side beyond what `declare_kernel!` already provides.

---

## Built-in media

Three deterministic generators, no files needed:

```rust
// 256×256 black-and-white tiles, 32 px each
let data = builtin_checkerboard(256, 256);

// Solid red fill
let data = builtin_solid_color(256, 256, Rgba8::RED);

// Horizontal gradient from black to white
let data = builtin_gradient_h(256, 256, Rgba8::BLACK, Rgba8::WHITE);
```

---

## Comparison & diff engine

Two modes, same API:

| Mode | Function | Speed | When to use |
|------|----------|-------|-------------|
| CPU metrics | `compute_metrics()` | Fast (~ms for 4 MP) | Numeric report: MAE, RMSE, PSNR, pixel counts |
| GPU heatmap | `diff_heatmap_gpu()` | Slower (GPU round-trip) | Visual diff: spatial error map |
| CPU heatmap | `write_heatmap_png()` | Fast | Same visual output, no GPU dispatch |

Both heatmap paths use a **blackbody colormap** controlled by a smoothstep:

```
error → smoothstep(smooth_a, smooth_b) → colour
         │                               │
         0.0 (cold)                       black
         0.25                             dark blue
         0.50                             bright blue
         0.75                             orange
         1.0 (hot)                        white
```

The `smooth_a` and `smooth_b` parameters let you adjust sensitivity:
- Narrow range (e.g. 0.0–0.05): amplifies tiny differences
- Wide range (e.g. 0.0–1.0): linear contrast across full range
- High floor (e.g. 0.3–0.7): ignores small differences, focuses on large ones

```rust
let config = DiffConfig {
    tolerance_r: 0.01,   // per-channel threshold for "different" count
    tolerance_g: 0.01,
    tolerance_b: 0.01,
    tolerance_a: 0.01,
    smooth_a: 0.0,        // heatmap: error ≤ 0.0 → black
    smooth_b: 1.0,        // heatmap: error ≥ 1.0 → white
};
```

### Metrics reference

All metrics are computed from raw pixel differences, independent of tolerance
(tolerance only affects the "pixels different" count).

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| `pixels_total` | w × h | positive | Total pixel count |
| `pixels_equal` | count(pixel where all 4 channels ≤ tol) | 0…total | Pixels within tolerance |
| `pixels_different` | total − equal | 0…total | Pixels exceeding tolerance on any channel |
| `different_ratio` | different / total | 0…1 | Fraction of differing pixels |
| `mean_absolute_error` | Σ\|rendered − ref\| / (pixels × 4) | 0…1 | Average per-channel deviation |
| `max_absolute_error` | max(\|rendered − ref\|) | 0…1 | Worst single-channel error |
| `root_mean_square_error` | √(Σ(rendered − ref)² / (pixels × 4)) | 0…1 | Penalizes large errors more |
| `psnr` | 20·log₁₀(1/RMSE) | 0…∞ dB | Higher = closer to reference |

---

## Tutorial 1 — Basic: Your first test

You have an effect crate with a Slang kernel. You want to verify it renders
correctly on a GPU.

### Step 1: Add the `image` crate to dev-dependencies

In your effect's `Cargo.toml`:

```toml
[dev-dependencies]
prgpu = { version = "0.1", features = ["testing"] }
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
```

### Step 2: Create the test directory

```
mkdir -p tests/assets tests/output
echo "/assets/" > tests/.gitignore
echo "/output/" >> tests/.gitignore
```

### Step 3: Write the test

```rust
// tests/render_basic.rs
use prgpu::testing::{GpuContext, builtin_checkerboard, write_png};
use my_effect::kernel::{my_kernel, MyParams};

#[test]
fn render_checkerboard_default() {
    let gpu = GpuContext::create().expect("GPU not available");

    let (w, h) = (512, 512);
    let input = builtin_checkerboard(w, h);

    let (in_buf, out_buf) = gpu.create_io_buffers(w, h, 4).expect("buffers");
    gpu.upload_to_buffer(&in_buf, &input, w, h, 4).expect("upload");

    let config = gpu.build_config(&in_buf, &out_buf, w, h, 4);

    let params = MyParams::default();
    unsafe { my_kernel(&config, params).expect("GPU kernel") };

    let output = gpu.download_from_buffer(&out_buf, w, h, 4).expect("download");

    assert!(!output.is_empty());
    let has_pixels = output.iter().any(|&b| b != 0);
    assert!(has_pixels, "output is all black — kernel may not have run");

    write_png("tests/output/basic.png", &output, w, h, 4).expect("write");
}
```

### Step 4: Run

```bash
cargo test -p my-effect --test render_basic -- --nocapture
```

If you have a GPU, the test renders and writes `tests/output/basic.png`.
Open it to inspect visually.

---

## Tutorial 2 — Advanced: Adding reference comparison

Visual inspection works for smoke tests, but you need automated pass/fail for CI.

### Step 1: Generate a golden reference

Run your effect once with known-good parameters, save the output as a reference:

```bash
# After running the basic test above:
cp tests/output/basic.png tests/assets/reference.png
```

### Step 2: Compare rendered output against the reference

```rust
use prgpu::testing::{
    GpuContext, builtin_checkerboard, DiffConfig,
    compute_metrics, write_heatmap_png, write_report_json, write_report_txt, write_png,
};

#[test]
fn render_against_reference() {
    let gpu = GpuContext::create().expect("GPU");

    // … render as before, get `output` …

    write_png("tests/output/actual.png", &output, w, h, 4).unwrap();

    // Load the golden reference
    let (reference, rw, rh) = load_image("tests/assets/reference.png");
    assert_eq!((rw, rh), (w, h));

    let config = DiffConfig {
        tolerance_r: 0.02,
        tolerance_g: 0.02,
        tolerance_b: 0.02,
        tolerance_a: 0.02,
        ..DiffConfig::default()
    };

    let report = compute_metrics(&output, &reference, w, h, &config).unwrap();

    // Fail if more than 1% of pixels exceed tolerance.
    // This catches rendering regressions without false positives from
    // minor floating-point differences.
    assert!(
        report.different_ratio < 0.01,
        "too many pixels differ: {:.2}%", report.different_ratio * 100.0
    );

    // Always produce diagnostics so you can investigate failures.
    write_heatmap_png("tests/output/heatmap.png", &output, &reference, w, h, &config).unwrap();
    write_report_txt("tests/output/report.txt", &report).unwrap();
    write_report_json("tests/output/report.json", &report).unwrap();
}
```

### Step 3: Interpret the heatmap

The heatmap shows **where** your effect diverges from the reference:

- **Black**: pixels are identical — the effect rendered correctly here.
- **Dark blue to bright blue**: tiny differences — likely floating-point noise,
  not a real problem.
- **Orange**: moderate differences — your effect changed something here.
  Worth investigating if unexpected.
- **White**: large differences — something is very wrong with these pixels.

A good render against its own reference should be nearly all black/blue.
If you see orange or white patches, zoom in on those areas in the actual vs
reference PNGs to understand what changed.

### Step 4: Custom input images

To test with real-world images instead of a checkerboard, load them:

```rust
fn load_bgra(path: &str) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap().to_rgba8();
    let (w, h) = (img.width(), img.height());
    let mut bgra = img.into_raw();
    for chunk in bgra.chunks_exact_mut(4) {
        chunk.swap(0, 2); // RGBA → BGRA
    }
    (bgra, w, h)
}

let (input, w, h) = load_bgra("tests/assets/hill.jpg");
```

### Step 5: Cross-tint testing

Test that your effect actually CHANGES the image (not just passes through):

```rust
// Render with blue tint
let blue_output = render_tint(&gpu, &input, w, h, /* tint=blue */);

// Generate a red-tinted reference (via Python/Pillow, or another effect)
let (red_ref, rw, rh) = load_bgra("tests/assets/hill_red_tint.png");

let report = compute_metrics(&blue_output, &red_ref, w, h, &DiffConfig::default()).unwrap();
// Blue vs red: should differ on most pixels.
assert!(report.pixels_different > report.pixels_total / 2);
assert!(report.mean_absolute_error > 0.05);
```

---

## Tutorial 3 — Statistical: Understanding the metrics

### What tolerance actually does

Tolerance is a **per-channel threshold** that decides whether a pixel is
counted as "equal" or "different" in `pixels_equal` / `pixels_different`.

It does **not** change the raw error values (MAE, max error, RMSE, PSNR).
Those are always computed from the actual pixel differences.

**Example — checkerboard with 50% blue tint:**

| Tolerance | Equal | Different | MAE | Max Error | RMSE |
|-----------|-------|-----------|-----|-----------|------|
| 0.00 | 0 | 65536 | 0.188 | 0.502 | 0.307 |
| 0.01 | 0 | 65536 | 0.188 | 0.502 | 0.307 |
| 0.25 | 0 | 65536 | 0.188 | 0.502 | 0.307 |
| **0.50** | **32768** | **32768** | 0.188 | 0.502 | 0.307 |
| 1.00 | 65536 | 0 | 0.188 | 0.502 | 0.307 |

Notice: MAE, Max Error, and RMSE are **constant** across all tolerances.
Only the "equal/different" count changes. At 0.50 tolerance, exactly half
the pixels cross the threshold — the other half are below.

This is because our checkerboard has only two colours after tinting:
- Black tiles → blue channel shift of 0.502
- White tiles → red/green channel shift of 0.498

At tolerance 0.50: 0.498 < 0.50 (pass), but 0.502 > 0.50 (fail).
Half pass, half fail — a clean binary split at the exact midpoint.

### MAE vs RMSE: which one matters?

`MAE` (Mean Absolute Error) answers: "On average, how wrong is each channel?"

`RMSE` (Root Mean Square Error) answers: "How wrong are the worst pixels?"

Because RMSE squares the error before averaging, it penalizes large
deviations more heavily than small ones.

**Example:**

| Pixel | Rendered | Reference | Abs Error | Sq Error |
|-------|----------|-----------|-----------|----------|
| A | 0.50 | 0.50 | 0.00 | 0.00 |
| A | 0.50 | 0.50 | 0.00 | 0.00 |
| B | 0.50 | 0.50 | 0.00 | 0.00 |
| B | 1.00 | 0.00 | 1.00 | 1.00 |

MAE = (0+0+0+0+0+0+1+1)/8 = 0.25
RMSE = √((0+0+0+0+0+0+1+1)/8) = √(2/8) = 0.50

Most pixels are perfect (6 of 8 channels at zero error), but one pixel has
a single channel blown out. MAE stays low (0.25), but RMSE jumps to 0.50
because the squared error of 1.0 dominates the average.

**When to use which:**
- MAE: when you care about overall quality. "Is the image generally close?"
- RMSE: when large errors are unacceptable. "Are there any glaring mistakes?"
- Both: for a complete picture. Low MAE + low RMSE = clean render.
  Low MAE + high RMSE = mostly fine with a few bad spots.

### PSNR: the decibel scale

PSNR (Peak Signal-to-Noise Ratio) is just RMSE converted to decibels:

```
PSNR = 20 × log₁₀(1 / RMSE)
```

It's a logarithmic scale — each +6 dB means the error has been halved.

| PSNR | Visual quality | Typical scenario |
|------|---------------|------------------|
| ∞ | Pixel-perfect | Rendered against identical reference |
| > 50 dB | Indistinguishable | Well within floating-point tolerance |
| 40–50 dB | Excellent | Minor rounding differences |
| 30–40 dB | Good | Slight colour shifts |
| 20–30 dB | Noticeable | Visible effect applied |
| 10–20 dB | Very different | Wrong tint, wrong strength |
| < 10 dB | Completely different | Different effect or broken render |

Our checkerboard tint example: RMSE = 0.307 → PSNR = 10.27 dB.
This is expected — applying a 50% blue tint SHOULD change the image
significantly.

The cross-tint test (blue vs red): RMSE = 0.353 → PSNR = 9.03 dB.
Even lower because blue and red are colour opposites.

### Asymmetric tolerance: per-channel sensitivity

Sometimes you want to be strict on luminance but lenient on colour:

```rust
let config = DiffConfig {
    tolerance_r: 0.001,  // red: very strict
    tolerance_g: 0.001,  // green: very strict
    tolerance_b: 0.500,  // blue: lenient (effect tints in blue)
    tolerance_a: 0.001,  // alpha: very strict
    ..DiffConfig::default()
};
```

This means: "I expect the red, green, and alpha channels to match exactly
(within 0.1%), but the blue channel can vary by up to 50%."

Our checkerboard + blue tint test shows this is still 100% different because:
- Black tiles fail on blue (0.502 > 0.500)
- White tiles fail on red/green (0.498 > 0.001)

There is no configuration that would produce a partial pass because the
errors are uniform and simultaneous across all channels.

### Smoothstep: controlling heatmap sensitivity

The heatmap maps error magnitude through a smoothstep sigmoid:

```
t = smoothstep(smooth_a, smooth_b, max_error)
```

`smooth_a` and `smooth_b` define the transition zone:
- Below `smooth_a`: error is treated as zero → black
- Above `smooth_b`: error is treated as maximum → white
- Between them: sigmoid ramp through the blackbody colours

**Practical examples:**

```rust
// Amplify tiny differences (0–5% error maps to full colormap range)
DiffConfig { smooth_a: 0.0, smooth_b: 0.05, ..Default::default() }

// Focus on significant differences (ignore everything below 30%)
DiffConfig { smooth_a: 0.3, smooth_b: 1.0, ..Default::default() }

// Narrow band around 0.5 (highlight which pixels are at exactly 50% error)
DiffConfig { smooth_a: 0.45, smooth_b: 0.55, ..Default::default() }
```

### Statistical test: monotonicity

A good diff algorithm must be **monotonic**: as tolerance increases, the
fraction of "different" pixels must never increase. Our `diff_monotonicity`
test verifies this property empirically across 7 tolerance levels.

The test also verifies that MAE, Max Error, and RMSE are **tolerance-invariant**
— they measure raw error, not pass/fail counts, so they must be constant
regardless of tolerance.

---

## GPU diff kernel internals

The `diff.slang` kernel follows PRGPU's 5-buffer convention:

| Buffer slot | Name | Role |
|------------|------|------|
| 0 | `outgoing` | Rendered image (effect output) |
| 1 | `incoming` | Reference image |
| 2 | `dst` | Heatmap output |
| 3 | `frame` | `FrameParams` (dimensions, descriptors) |
| 4 | `params` | `DiffParams` (tolerances + smoothstep bounds) |

The `DiffParams` struct must match byte-for-byte between Slang and Rust
(both sides use `_pad*` fields for vec4 alignment):

**Slang** (`diff.slang`):
```
float tolR; float tolG; float tolB; float tolA;
float smoothA; float smoothB;
uint _pad0; uint _pad1;   // → 8 × 4 = 32 bytes
```

**Rust** (`kernels/diff.rs`):
```rust
#[repr(C)]
pub struct DiffParams {
    pub tol_r: f32, pub tol_g: f32, pub tol_b: f32, pub tol_a: f32,
    pub smooth_a: f32, pub smooth_b: f32,
    pub _pad0: u32, pub _pad1: u32,
}
```

---

## Platform support

| Platform | Backend | GPU required | Status |
|----------|---------|-------------|--------|
| macOS (Apple Silicon) | Metal | Yes | ✅ |
| Windows (NVIDIA GPU) | CUDA | Yes | ✅ |
| Either without GPU | — | — | Test skipped cleanly |

Tests run on real GPU hardware. There is no CPU fallback — if the GPU is
unavailable, `GpuContext::create()` returns an error.

---

## Debug checklist

| Symptom | Likely cause |
|---------|-------------|
| `GpuContext::create()` error | No GPU or driver mismatch |
| `STATUS_ACCESS_VIOLATION` on CUDA | `nvcuda.dll` absent or wrong CUDA version |
| `playground()` returns error | Buffer sizes mismatched, null pointers, or PTX incompatible with GPU |
| Output is all black | Shader compiled but didn't write to `dst` (check binding indices) |
| `compute_metrics()` size mismatch | Input and reference have different dimensions or bit depth |
| Heatmap is uniform colour | Input image has uniform error (e.g., checkerboard tint). Test on a photo to see gradients. |
| GPU heatmap disagrees with CPU | Floating-point rounding in `TextureView.Load()` vs `/ 255.0` |
