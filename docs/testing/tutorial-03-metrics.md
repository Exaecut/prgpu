# Tutorial 3 — Metrics Explained

This tutorial explains every metric and configuration parameter the diff engine
produces. You'll learn what they mean, how they interact, and how to read them
to diagnose rendering problems. All numbers come from a real test: a 256×256
checkerboard passed through a 50% blue tint.

The metrics work identically whether you render through `HostBuilder` (full
Premiere path) or direct kernel dispatch — both produce the same pixel output
that `compute_metrics()` compares.

## What tolerance actually does

Tolerance is a **per-channel threshold** that decides whether a pixel is counted
as "equal" or "different." It does **not** change the raw error values (MAE, max
error, RMSE, PSNR) — those are always computed from the actual pixel differences.

Think of tolerance as a pass/fail gate. If every channel's error is below its
threshold, the pixel passes. If any channel exceeds, it fails. The tolerance you
set determines how strict that gate is.

**Checkerboard + 50% blue tint — tolerance sweep:**

| Tolerance | Equal | Different | MAE | Max Error | RMSE |
|-----------|-------|-----------|-----|-----------|------|
| 0.00 | 0 | 65536 | 0.188 | 0.502 | 0.307 |
| 0.01 | 0 | 65536 | 0.188 | 0.502 | 0.307 |
| 0.25 | 0 | 65536 | 0.188 | 0.502 | 0.307 |
| **0.50** | **32768** | **32768** | 0.188 | 0.502 | 0.307 |
| 1.00 | 65536 | 0 | 0.188 | 0.502 | 0.307 |

MAE, Max Error, and RMSE are constant across all tolerances. Only the
"equal/different" count changes. The transition happens at 0.50 because:

- **Black tiles**: the blue channel shifts by 0.502. At tolerance 0.50, 0.502 > 0.50 → **fail**.
- **White tiles**: the red and green channels shift by 0.498. At tolerance 0.50, 0.498 < 0.50 → **pass**.

Exactly 32768 of each. A clean binary split at the midpoint of the error range.

### Choosing a tolerance

- **0.00** — zero tolerance. Every pixel with any floating-point difference fails. Only use for exact-match verification.
- **0.01–0.02** — strict. Catches real rendering changes while ignoring sub-pixel rounding.
- **0.05–0.10** — moderate. Useful when you expect some variance (e.g., noise, dithering).
- **0.25–0.50** — lenient. Only catches large deviations.
- **1.00** — everything passes. Use when you only care about raw error metrics.

## MAE vs RMSE

These two metrics answer different questions about the same data.

**MAE** (Mean Absolute Error) = average per-channel deviation.

```
MAE = sum of all |rendered − reference| / (total_pixels × 4)
```

It answers: "On average, how wrong is each channel?" Every pixel contributes
linearly — a 0.5 error has exactly 5× the weight of a 0.1 error.

**RMSE** (Root Mean Square Error) penalises large errors more heavily.

```
RMSE = √( sum of all (rendered − reference)² / (total_pixels × 4) )
```

It answers: "How wrong are the worst pixels?" Because the error is squared
before averaging, a single large deviation dominates the result.

**Worked example** — two pixels, each with 4 channels (R, G, B, A):

| Pixel | Rendered | Reference | Abs Error | Sq Error |
|-------|----------|-----------|-----------|----------|
| A R | 0.50 | 0.50 | 0.00 | 0.00 |
| A G | 0.50 | 0.50 | 0.00 | 0.00 |
| A B | 0.50 | 0.50 | 0.00 | 0.00 |
| A A | 0.50 | 0.50 | 0.00 | 0.00 |
| B R | 0.50 | 0.50 | 0.00 | 0.00 |
| B G | 0.50 | 0.50 | 0.00 | 0.00 |
| B B | 1.00 | 0.00 | 1.00 | 1.00 |
| B A | 0.50 | 0.50 | 0.00 | 0.00 |

MAE  = (0+0+0+0+0+0+1+0) / 8 = 0.125
RMSE = √((0+0+0+0+0+0+1+0) / 8) = √(1/8) = 0.354

Most channels are perfect, but one channel is blown out. MAE stays low (0.125),
but RMSE jumps to 0.354 because the square of 1.0 dominates the average.

**When to use each:**
- **MAE** when you care about overall quality — "does the image generally look right?"
- **RMSE** when large errors are unacceptable — "are there any glaring mistakes?"
- **Both** for a complete picture. Low MAE + low RMSE = clean render. Low MAE + high RMSE = mostly fine with a few bad spots. High MAE + low RMSE = uniformly mediocre.

## PSNR — the decibel scale

PSNR converts RMSE to a logarithmic scale, like audio decibels:

```
PSNR = 20 × log₁₀(1 / RMSE)
```

It's logarithmic: each +6 dB means the error has been **halved**.

| PSNR | Visual quality | Typical scenario |
|------|---------------|------------------|
| ∞ | Pixel-perfect | Rendered against identical reference |
| > 50 dB | Indistinguishable | Within floating-point tolerance |
| 40–50 dB | Excellent | Minor rounding differences |
| 30–40 dB | Good | Slight colour shifts |
| 20–30 dB | Noticeable | Visible effect applied |
| 10–20 dB | Very different | Wrong tint or strength |
| < 10 dB | Completely different | Different effect or broken render |

Our checkerboard with 50% blue tint: RMSE 0.307 → PSNR 10.27 dB.
This is expected — applying a strong tint *should* change the image significantly.
If your effect produces PSNR > 30 against an unmodified input, it may be a no-op.

The cross-tint test (blue GPU vs red Python reference, hill.jpg): RMSE 0.353 →
PSNR 9.03 dB. Even lower because blue and red are colour opposites.

## Asymmetric tolerance — per-channel sensitivity

All four channels don't need the same strictness. If your effect only
operates on the blue channel, you can be looser on blue and tighter on red/green:

```rust
let config = DiffConfig {
    tolerance_r: 0.001,  // red: very strict — almost zero tolerance
    tolerance_g: 0.001,  // green: very strict
    tolerance_b: 0.500,  // blue: lenient — the effect tints in blue
    tolerance_a: 0.001,  // alpha: very strict
    ..DiffConfig::default()
};
```

This means: "Red, green, and alpha must match within 0.1%. Blue can vary by up to 50%."

On our checkerboard test with a 50% blue tint, this configuration still produces
100% different pixels because the errors hit all channels simultaneously:
- Black tiles fail on blue (0.502 > 0.500)
- White tiles fail on red/green (0.498 > 0.001)

Asymmetric tolerance is most useful when your effect operates on a subset of
channels (e.g., a luma-only filter that shouldn't touch chroma, or an alpha
matte generator).

## Smoothstep — controlling the heatmap visually

The blackbody heatmap maps error magnitude through a smoothstep sigmoid:

```
t = smoothstep(smooth_a, smooth_b, max_error)
```

where `t = 0` maps to black and `t = 1` maps to white, with the thermal
gradient in between (blue → orange → white).

`smooth_a` and `smooth_b` define the transition zone:

| Configuration | Effect |
|--------------|--------|
| `smooth_a: 0.0, smooth_b: 0.05` | Amplify tiny differences — 0–5% error fills the full colormap. Useful for spotting subtle rounding artifacts. |
| `smooth_a: 0.0, smooth_b: 1.0` | Linear contrast — 0–100% error maps linearly across the full gradient. Good default. |
| `smooth_a: 0.3, smooth_b: 1.0` | Ignore errors below 30%. Only significant deviations produce colour. |
| `smooth_a: 0.45, smooth_b: 0.55` | Narrow band around 50%. Highlights which pixels are at exactly that error level — useful for verifying a specific tint strength. |

The smoothstep is applied in both the GPU kernel (`diff.slang`) and the CPU
path (`write_heatmap_png`), so results are consistent regardless of which path
you use.

## Monotonicity — a property your diff engine must have

A correct diff algorithm is **monotonic**: as you increase tolerance, the number
of pixels counted as "different" must never go up. If raising tolerance from
0.01 to 0.05 suddenly flags *more* pixels as different, the algorithm is flawed.

Our test suite verifies monotonicity across 7 tolerance levels (0.0 through 1.0)
and confirms that MAE, Max Error, and RMSE remain constant — they measure raw
error, not pass/fail counts, so they must be tolerance-invariant.

## Reading the JSON report

Every comparison produces a machine-readable JSON report:

```json
{
  "width": 2600,
  "height": 1548,
  "pixels_total": 4024800,
  "pixels_equal": 0,
  "pixels_different": 4024800,
  "different_ratio": 1.0,
  "mean_absolute_error": 0.25,
  "max_absolute_error": 0.514,
  "root_mean_square_error": 0.353,
  "psnr": 9.03,
  "thresholds": {
    "per_channel_absolute": [0.01, 0.01, 0.01, 0.01],
    "max_different_ratio": 0.01
  },
  "passed": true
}
```

Feed this to CI dashboards, regression tracking, or automated alerts. The
`passed` field reports whether thresholds were met (always `true` in report-only
mode; configurable via `DiffPolicy`).
