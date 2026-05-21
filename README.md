# prgpu

GPU-accelerated rendering utilities for Adobe Premiere Pro and After Effects
plugins, powered by [Slang](https://github.com/shader-slang/slang) shaders that
compile once and dispatch uniformly across Metal (macOS), CUDA (Windows), and a
Rust/rayon-based CPU fallback.

## What you get

- **Single shader, three targets.** Author compute shaders in Slang; prgpu's
  `build::compile_shaders("./shaders")` pipeline emits Metal `.metallib`, CUDA
  `.cu` / PTX, and C++ for the CPU path in one go. Constant buffers and
  struct layouts are verified byte-for-byte across all three.
- **Zero-cost dispatch abstraction.** `prgpu::declare_kernel!(name, Params)`
  generates typed entry points so kernel code in the host looks the same
  whether the dispatch lands on MTLComputeCommandEncoder, cuLaunchKernel, or a
  rayon tile loop.
- **CPU fallback that isn't a toy.** The `cpu::render` module batches work into
  row-strip tiles, sizes them to the host's rayon pool, and re-uses allocations
  across frames via a keyed buffer cache. Real production effects
  (radialblur, vignette, etc.) hit realtime-preview budgets at 1080p on the
  CPU path.
- **Mip-chain pyramids built in.** `pipeline::mip::prepare_mip_source` +
  `generate_mips` turn a flat input buffer into a multi-lod source in a couple
  of lines of host code. Effects whose sample radius grows with a parameter
  (radial blur, bokeh, glow, motion blur) can sample from downsampled levels
  for multi-x speedups with no user-visible quality loss. See
  [`docs/mip_chain.md`](docs/mip_chain.md) for the design.
- **Benchmarking out of the box.** With the `bench` feature, a kernel author
  gets `prgpu::bench::{Scene, Resolution, PixelFormat, KernelBenchmark}` — a
  criterion harness that runs the real CPU dispatch against synthetic frames,
  so you measure the kernel and nothing else.

## Crate layout

```
prgpu
├── src
│   ├── types/            Configuration, TextureDesc, FrameParams, Pixel
│   ├── cpu/              render_cpu / render_cpu_direct, tile scheduler,
│   │                     bounded rayon pool, keyed buffer cache
│   ├── gpu/              backend dispatch (Metal / CUDA), hot-reload hooks,
│   │                     device-agnostic buffer cache
│   ├── kernels/          declare_kernel!, built-in mip_downsample, helpers
│   │                     for common patterns (blur downsample, sweep samples)
│   ├── params/           FromParam, SetupParams, CpuParams traits + kernel
│   │                     params DSL macros
│   ├── ui/               BlendMode popup, shared UI parameter helpers
│   ├── build/            slangc driver, reflection-driven Rust binding gen,
│   │                     cpp → static lib compile (feature = "build")
│   └── bench.rs          criterion harness (feature = "bench")
├── shaders/              built-in Slang shaders (mip_downsample, etc)
├── docs/                 design notes: mip_chain, timing, xframe
└── prgpu-macro/          #[gpu_struct] proc-macro crate (transitive dep)
```

## Usage

```toml
# Cargo.toml of a Premiere / AE plugin crate
[dependencies]
prgpu = "0.1"

[build-dependencies]
prgpu = { version = "0.1", features = ["build"] }

[dev-dependencies]
prgpu = { version = "0.1", features = ["timing", "bench"] }
```

```rust
// build.rs
fn main() {
    prgpu::build::compile_shaders("./shaders").unwrap();
    // ...
}
```

```rust
// src/kernel.rs
prgpu::kernel_params! {
    MyParams for crate::params::Params {
        strength: f32 = [float(Strength) / 100.0];
    }
}
prgpu::declare_kernel!(my_effect, MyParams);
```

## Features

| Feature             | Enables                                                   |
|---------------------|-----------------------------------------------------------|
| `default`           | `cargo-clippy` (a dummy target-less feature)              |
| `timing`            | `timing::log_snapshot()` instrumentation                  |
| `bench`             | `prgpu::bench::*` criterion harness                       |
| `build`             | `prgpu::build::compile_shaders` slangc driver (build-deps)|
| `shader_hotreload`  | NVRTC runtime recompilation on Windows                    |

## Slang SDK

prgpu's `build` feature auto-downloads the Slang SDK into
`target/.slang-sdk/<version>/` on first build. No manual setup. The version
is pinned in `src/build/sdk.rs`.

## Shader include path (vekl or your own)

When you call `compile_shaders("./shaders")`, prgpu looks for a sibling
directory named `vekl` (relative to prgpu's own checkout) and adds it to the
slangc include path if found. If your effect crate isn't laid out next to
a `vekl/` checkout, call the explicit form instead:

```rust
use std::path::PathBuf;
prgpu::build::compile_shaders_with(
    "./shaders",
    &[PathBuf::from("path/to/your/shader-lib")],
).unwrap();
```

prgpu is happy to run without any extra include path — your shaders just have
to resolve their own `import` / `#include` statements against the directories
you pass.

## Documentation

- [docs/mip_chain.md](docs/mip_chain.md) — host-side pyramid blur recipe
- [docs/timing.md](docs/timing.md) — instrumentation model
- [docs/xframe.md](docs/xframe.md) — cross-frame state contract

## License

Dual-licensed under MIT OR Apache-2.0.

## Status

APIs are stabilising; the `0.1.x` series may still see breaking changes in
minor bumps. Pinning an exact `=0.1.N` version is reasonable if you're
shipping a release binary.
