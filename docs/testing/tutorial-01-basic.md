# Tutorial 1 — Basic: Your First GPU Test

You'll write a minimal test that renders a checkerboard through the full Premiere
render chain on the real GPU, downloads the output, and writes a PNG for visual
inspection.

This tutorial uses the **HostBuilder** path — the recommended approach that
exercises `PremiereGPU::render()` exactly as Premiere Pro would call it.

## Prerequisites

Your effect must expose its `GpuFilter` struct and param enum publicly:

```rust
// src/lib.rs
pub mod kernel;
pub mod gpu;       // make public (host test needs PremiereGPU)
pub mod params;    // make public (host test needs Params enum)

// src/gpu.rs
#[derive(Default)]
pub struct PremiereGPU;   // make public
```

The `declare_kernel!` macro in `kernel.rs` is already public — no changes needed.

## Step 1 — Dependencies

Add to your effect's `Cargo.toml`:

```toml
[dev-dependencies]
prgpu = { version = "0.1", features = ["testing"] }
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
```

## Step 2 — Create the test directory

```bash
mkdir -p tests/assets tests/output
echo "/assets/"  > tests/.gitignore
echo "/output/" >> tests/.gitignore
```

## Step 3 — Write the test

Create `tests/render_basic.rs`:

```rust
use prgpu::testing::{HostBuilder, ParamValue, builtin_checkerboard, write_png};
use my_effect::gpu::PremiereGPU;
use my_effect::params::Params;

#[test]
fn render_checkerboard_tint() {
    let (w, h) = (512, 512);
    let input = builtin_checkerboard(w, h);

    let ctx = HostBuilder::<PremiereGPU, Params>::new(PremiereGPU, input, w, h)
        .param(Params::Strength, ParamValue::float(50.0))
        .param(Params::Tint, ParamValue::color(0, 0, 255, 255))
        .param(Params::ExpandFrame, ParamValue::bool(false))
        .build()
        .expect("HostContext");

    let output = ctx.start().expect("render chain");

    assert!(!output.is_empty());
    let has_pixels = output.iter().any(|&b| b != 0);
    assert!(has_pixels, "output is all black — kernel may not have run");

    write_png("tests/output/basic.png", &output, w, h, 4).expect("write");
}
```

## Step 4 — Run

```bash
cargo test -p my-effect --test render_basic -- --nocapture
```

Open `tests/output/basic.png`. If it looks correct, your GPU pipeline works.

## What happens when `.start()` runs

`.start()` executes the full Premiere render chain in order:

1. **`PremiereGPU::global_init()`** — initialises global CUDA/Metal state.
2. **`PremiereGPU::get_frame_dependencies()`** — queries frame dependency info.
3. **`PremiereGPU::precompute()`** — precomputation step (no-op for most effects).
4. **`PremiereGPU::render()`** — the main render:
   - `GPURenderProperties::new()` — builds render properties from mock `GpuFilterData` and `PPixHand`.
   - `YourParams::from_gpu()` — extracts user parameters from mock `VideoSegmentSuite`.
   - `Configuration::effect()` — builds the GPU buffer configuration.
   - `your_kernel(&config, params)` — dispatches the real Slang kernel on the GPU.
5. **`PremiereGPU::global_destroy()`** — cleans up GPU resources.

All mock Premiere objects (suites, `GpuFilterData`, `RenderParams`, `PPix`)
are constructed in heap memory with real function pointers, so the render path
is identical to what Premiere Pro triggers.

## Alternative: Direct kernel dispatch

For quick smoke tests that don't need the full Premiere chain, use `GpuContext`
directly:

```rust
use prgpu::testing::{GpuContext, builtin_checkerboard, write_png};
use my_effect::kernel::{my_kernel, MyParams};

let gpu = GpuContext::create().expect("GPU");
let (w, h) = (512, 512);
let input = builtin_checkerboard(w, h);

let (in_buf, out_buf) = gpu.create_io_buffers(w, h, 4).expect("buffers");
gpu.upload_to_buffer(&in_buf, &input, w, h, 4).expect("upload");

let config = gpu.build_config(&in_buf, &out_buf, w, h, 4);
let params = MyParams::default();
unsafe { my_kernel(&config, params).expect("kernel") };

let output = gpu.download_from_buffer(&out_buf, w, h, 4).expect("download");
write_png("tests/output/direct.png", &output, w, h, 4).expect("write");
```

This bypasses the mock FFI layer — faster to iterate, but doesn't exercise
`from_gpu()` or `Configuration::effect()`. Use HostBuilder for CI and
regression tests.

## Next

> **Tutorial 2 — Advanced**: Replace the manual visual check with automated
> reference comparison. Add a golden image, generate heatmaps that show
> *where* and *how much* your render diverges, load custom input photos,
> and set up cross-tint tests to verify your effect actually changes the image.

[→ Tutorial 2 — Advanced](tutorial-02-advanced.md)
