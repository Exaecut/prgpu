# Tutorial 1 — Basic: Your First GPU Test

You'll create a minimal test that loads a checkerboard, dispatches your Slang
kernel on the GPU, downloads the rendered pixels, and writes a PNG for visual
inspection. No reference comparison yet — just verifying the pipeline works
end-to-end.

## Prerequisites

Your effect crate must use `declare_kernel!` in its `kernel.rs`. The macro
already generates a public GPU dispatch function — tests import it directly.
No extra code is needed in the effect.

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

## Step 4 — Run

```bash
cargo test -p my-effect --test render_basic -- --nocapture
```

Open `tests/output/basic.png`. If it looks correct, your GPU pipeline works.

## What each line does

- **`GpuContext::create()`** — initialises Metal (macOS) or CUDA (Windows). Returns an error if no GPU is found.
- **`builtin_checkerboard(w, h)`** — generates a 512×512 BGRA checkerboard with 32 px tiles. No external file needed.
- **`create_io_buffers(w, h, 4)`** — allocates one input buffer and one output buffer on the GPU through PRGPU's LRU cache. `4` means BGRA8.
- **`upload_to_buffer()`** — copies your tightly-packed BGRA pixel data to the GPU buffer via a staging copy.
- **`build_config()`** — assembles a `Configuration` with GPU device handles, buffer pointers, pitches, and the BGRA pixel layout convention.
- **`my_kernel(&config, params)`** — dispatches your real Slang compute shader on the GPU via the generated `declare_kernel!` entry point.
- **`download_from_buffer()`** — reads the GPU output buffer back to a `Vec<u8>` on the host.
- **`write_png()`** — swizzles BGRA to RGBA and saves a PNG through the `image` crate.

## Next

> **Tutorial 2 — Advanced**: Replace the manual visual check with automated
> reference comparison. Add a golden image, generate heatmaps that show
> *where* and *how much* your render diverges, load custom input photos,
> and set up cross-tint tests to verify your effect actually changes the image.

[→ Tutorial 2 — Advanced](tutorial-02-advanced.md)
