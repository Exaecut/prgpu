# Tutorial 2 — Advanced: Reference Comparison & Heatmaps

You'll replace manual visual inspection with automated reference comparison.
A golden image acts as the ground truth; the harness computes numeric metrics
and generates a heatmap showing *where* and *how much* your render diverged.

## Step 1 — Create a golden reference

Run your basic test once, then copy the output as your reference:

```bash
cargo test -p my-effect --test render_basic -- --nocapture
cp tests/output/basic.png tests/assets/reference.png
```

This reference represents "correct behaviour." Future test runs will compare
against it — if anything changes, the test fails.

## Step 2 — Write the comparison test

```rust
use prgpu::testing::{
    HostBuilder, ParamValue, DiffConfig, builtin_checkerboard,
    compute_metrics, write_heatmap_png, write_report_json, write_report_txt, write_png,
};
use my_effect::gpu::PremiereGPU;
use my_effect::params::Params;

fn load_bgra(path: &str) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap().to_rgba8();
    let (w, h) = (img.width(), img.height());
    let mut bgra = img.into_raw();
    for chunk in bgra.chunks_exact_mut(4) {
        chunk.swap(0, 2); // RGBA → BGRA
    }
    (bgra, w, h)
}

#[test]
fn render_against_reference() {
    let (w, h) = (512, 512);
    let input = builtin_checkerboard(w, h);
    let input_copy = input.clone();

    let ctx = HostBuilder::<PremiereGPU, Params>::new(PremiereGPU, input, w, h)
        .param(Params::Strength, ParamValue::float(50.0))
        .param(Params::Tint, ParamValue::color(0, 0, 255, 255))
        .param(Params::ExpandFrame, ParamValue::bool(false))
        .build()
        .expect("HostContext");

    let output = ctx.start().expect("render chain");
    write_png("tests/output/actual.png", &output, w, h, 4).unwrap();

    let (reference, rw, rh) = load_bgra("tests/assets/reference.png");
    assert_eq!((rw, rh), (w, h), "reference dimensions mismatch");

    let config = DiffConfig {
        tolerance_r: 0.02,
        tolerance_g: 0.02,
        tolerance_b: 0.02,
        tolerance_a: 0.02,
        ..DiffConfig::default()
    };

    let report = compute_metrics(&output, &reference, w, h, &config).unwrap();

    // Fail if more than 1% of pixels differ — catches regressions
    // without false positives from minor floating-point noise.
    assert!(
        report.different_ratio < 0.01,
        "too many pixels differ: {:.2}%", report.different_ratio * 100.0
    );

    // Diagnostic outputs for investigating failures
    write_heatmap_png("tests/output/heatmap.png", &output, &reference, w, h, &config).unwrap();
    write_report_txt("tests/output/report.txt", &report).unwrap();
    write_report_json("tests/output/report.json", &report).unwrap();
}
```

Note: `input_copy` lets you compare `output` against the original input after
`HostBuilder` takes ownership of `input`. Clone before passing to the builder
if you need the original data afterwards.

## Step 3 — Interpret the heatmap

The heatmap uses a thermal colormap — **black → dark blue → bright blue → orange → white**
— to encode per-pixel error magnitude:

| Colour | Meaning |
|--------|---------|
| **Black** | Pixel is identical to reference — correct. |
| **Dark blue** | Tiny difference, likely floating-point noise. Ignore. |
| **Bright blue** | Small but measurable difference. Worth a glance. |
| **Orange** | Moderate difference — your effect changed something here. Investigate if unexpected. |
| **White** | Large difference — something is very wrong with this pixel. |

A healthy test against its own reference should be nearly all black and dark blue.
Orange or white patches mean something diverged — compare `actual.png` against
`reference.png` in those regions.

## Step 4 — Custom input images

Checkerboards are fast but unrealistic. Test with photos:

```rust
let (input, w, h) = load_bgra("tests/assets/hill.jpg");
```

Place `.jpg` or `.png` files in `tests/assets/`. The `load_bgra` helper above
handles RGBA→BGRA swizzle and returns `(data, width, height)`.

Run your effect on real images to catch edge cases that a checkerboard misses:
high-frequency detail, smooth gradients, skin tones, dark areas.

## Step 5 — Cross-tint: verify your effect actually works

A test that compares an effect against its own reference can pass trivially if
the effect is a no-op. Cross-tint testing verifies the effect *does* change
pixels:

```rust
// Render with blue tint via HostBuilder
let ctx = HostBuilder::<PremiereGPU, Params>::new(PremiereGPU, input, w, h)
    .param(Params::Strength, ParamValue::float(50.0))
    .param(Params::Tint, ParamValue::color(0, 0, 255, 255))
    .build()?;
let blue_output = ctx.start()?;

// Generate a red-tinted reference (Python/Pillow, or a separate effect pass)
let (red_ref, rw, rh) = load_bgra("tests/assets/hill_red_tint.png");

let report = compute_metrics(&blue_output, &red_ref, w, h, &DiffConfig::default()).unwrap();

// Blue vs red should differ on most pixels.
assert!(report.pixels_different > report.pixels_total / 2);
assert!(report.mean_absolute_error > 0.05);
```

This catches bugs where parameters are ignored and the effect becomes a
pass-through. If blue tint vs red tint shows 0 different pixels, something
is broken.

## Next

> **Tutorial 3 — Metrics Explained**: Deep dive into every metric the diff
> engine produces. Learn what tolerance actually does, when to use MAE vs
> RMSE, how to read PSNR, configure asymmetric per-channel thresholds, and
> control heatmap sensitivity with smoothstep — all with real numbers from
> our test suite on a checkerboard + blue tint.

[→ Tutorial 3 — Metrics Explained](tutorial-03-metrics.md)
