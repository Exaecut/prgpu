# Mip chain — host-side guide

How to opt a prgpu-powered effect into a mip-chain source buffer, what
the allocator does differently between backends, and the exact host
code pattern each effect uses.

Shader-side reference: [`vekl/docs/reference/texture/descriptor.md`](../../vekl/docs/reference/texture/descriptor.md)
and [`vekl/docs/reference/texture/view.md`](../../vekl/docs/reference/texture/view.md).

Tutorial with a working slang shader: [`vekl/docs/tutorials/01-mip-chain-pyramid-blur.md`](../../vekl/docs/tutorials/01-mip-chain-pyramid-blur.md).

---

## What it gives you

Sampling a variable-size neighborhood (motion blur, zoom blur, glow,
bokeh, depth-of-field) gets **flat cost** regardless of radius. The
kernel picks a mip level based on the local filter size and reads a
fixed number of taps — moving from lod 0 to lod 2 cuts the underlying
compute by 16× while producing a visually equivalent result. The
Phase 3 radialblur in this repo uses this pattern to drop 1080p CPU
frame times from ~850 ms (adaptive N, Phase 1) to a target of
<150 ms.

## When to enable it

Only when the shader actually benefits: the mip generation itself
costs one kernel dispatch per extra level (roughly `(4/3) × 1` texel
writes for the whole chain — a couple of milliseconds at 1080p).
Effects that read a single pixel per output never need mips; effects
that read a growing neighborhood (radius / kernel size driven by a
user parameter) usually do.

- Effect reads 1 source texel per output → don't enable.
- Effect reads a small fixed kernel (3×3 / 5×5) → don't enable.
- Effect reads a radius-dependent neighborhood (blur / glow / bokeh
  / sweep) → enable with `outgoing_mip_levels >= 2`.

## Constants

- `prgpu::types::MAX_MIP = 5` — upper bound on the chain depth. Must
  match `vekl::MAX_MIP`. Five levels cover 1/16 per axis which is
  enough for any sweep blur pyramid.
- Mip chain is opt-in per dispatch via
  `Configuration::outgoing_mip_levels` (default `0` = disabled).
  `0` or `1` means "no mip chain"; values `2..=MAX_MIP` request an
  N-level pyramid.

## Buffer sizing

Every level below lod 0 is tightly packed (no row padding), so the
total byte budget stays under `ceil(4/3) × base_bytes`:

| Level | Dims            | Bytes (bgra8) |
|-------|-----------------|---------------|
| 0     | W × H           | W × H × 4     |
| 1     | W/2 × H/2       | (W/2) × (H/2) × 4 |
| 2     | W/4 × H/4       | (W/4) × (H/4) × 4 |
| ...   | ...             | ...           |

The helper `prgpu::types::mip_buffer_size_bytes(w, h, bpp, levels)`
returns the exact byte count needed. Every allocator that takes a
`mip_levels` parameter internally calls this.

---

## The two-call recipe

Every effect that opts into mips does exactly this:

```rust
config.outgoing_mip_levels = 4;
let _mip_buf = unsafe {
    prgpu::kernels::mip::prepare_mip_source(&mut config, MY_MIP_TAG)?
};
unsafe { prgpu::kernels::mip::generate_mips(&config)?; }
unsafe { effect_kernel(&config, user_params)?; }
```

`prepare_mip_source` handles all three backend-specific steps
internally:

1. Allocates a mip-capable buffer via the backend allocator (CPU `Vec`
   cache, Metal `MTLBuffer`, or CUDA device memory).
2. Copies the current `config.outgoing_data` into level 0 of that
   buffer using the backend-native copy primitive (`std::ptr::
   copy_nonoverlapping` / `MTLBlitCommandEncoder` / `cuMemcpy2D_v2`),
   handling mismatched row pitches when Premiere's source has padding
   rows that the tight mip buffer doesn't.
3. Swaps `config.outgoing_data` to the new buffer so subsequent
   dispatches (mip gen + effect kernel) read through it.

The returned `ImageBuffer` keeps the allocation alive in the prgpu
cache — the caller binds it to `_` so the borrow checker keeps it in
scope for the remainder of the frame. Dropping it has no GPU side
effect; the cache owns the allocation and reuses it across frames.

`generate_mips` then runs the prgpu built-in downsampler N-1 times to
fill levels 1..N-1. Both functions are cheap no-ops when
`outgoing_mip_levels <= 1`, so effects can unconditionally call them
when the user's quality knob sits at zero.

---

## CPU vs GPU mip chain

The **shader** is identical — it reads `frame.outDesc.mipLevelCount /
mipOffsetBytes / mipWidth / mipHeight / mipPitchBytes` and calls
`TextureView::SampleLinear(uv, lod)` / `SampleLinearTrilinear(uv, lodF)`
the same way on all backends. VEKL maps pixel I/O to a byte-addressed
`StructuredBuffer<uint>` on CPU, Metal, and CUDA alike, so the mip
offsets resolve identically.

The **host** side differs in three specific places:

### 1. Buffer allocator

| Backend | Function | Returns |
|---------|----------|---------|
| CPU     | `prgpu::cpu::buffer::get_or_create_with_mips(w, h, bpp, levels, tag)` | `ImageBuffer { buf: BufferObj { raw: *mut c_void }, .. }` backed by a thread-local `Vec<u8>` cache |
| Metal   | `unsafe { prgpu::gpu::backends::metal::buffer::get_or_create_with_mips(device, w, h, bpp, levels, tag) }` | `ImageBuffer` whose `buf.raw` is a `*mut Object` pointing at a Metal `MTLBuffer` (private storage) |
| CUDA    | `unsafe { prgpu::gpu::backends::cuda::buffer::get_or_create_with_mips(device, w, h, bpp, levels, tag) }` | `ImageBuffer` whose `buf.raw` is a `CUdeviceptr` (cast to `*mut c_void`) from `cuMemAlloc_v2` |

All three allocators share a 12-slot LRU cache keyed on `(w, h, bpp,
mip_levels, tag, device)` (the `device` field is ignored on CPU).
Requesting the same buffer twice in a frame returns the cached one;
on eviction the old buffer is released (Metal `release`, CUDA
`cuMemFree_v2`, CPU `Vec::drop`).

### 2. Copying level 0 in

The allocator returns an uninitialised buffer — `prepare_mip_source`
handles the copy for you. Under the hood it picks the right primitive
for the backend:

| Backend | Copy primitive |
|---------|----------------|
| CPU     | `std::ptr::copy_nonoverlapping(src, dst, row_bytes × height)` when pitches match; row-by-row copies otherwise. |
| Metal   | `MTLBlitCommandEncoder` `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` — a single flat blit when pitches match, otherwise one blit per row. Command buffer is submitted + awaited inside `prepare_mip_source` so `generate_mips` sees the copied data. |
| CUDA    | `cuMemcpyDtoD_v2(dst, src, size)` when pitches match, `cuMemcpy2D_v2` (with explicit src/dst pitch) when they don't. Synchronous on the default stream. |

Calling the helper yourself is only useful if you need a custom
copy path (e.g. color-converting the source before the pyramid).
Otherwise `prepare_mip_source` is all you need.

### 3. `generate_mips` dispatch routing

`prgpu::kernels::mip::generate_mips(&config)` is a thin wrapper around
N-1 dispatches of the built-in `mip_downsample` compute kernel. It
routes automatically:

```rust
// From prgpu/src/kernels/mip.rs (simplified):
if !pass_cfg.device_handle.is_null() {
    // GPU path: Metal / CUDA via dispatch_kernel
    unsafe { mip_downsample(&pass_cfg, params)? };
} else {
    // CPU path: pure rayon tile loop, no AE plumbing
    unsafe {
        crate::cpu::render::render_cpu_direct(
            "mip_downsample", &pass_cfg,
            MIP_DOWNSAMPLE_CPU_DISPATCH_TILE, &params,
        );
    }
}
```

- **GPU (Metal / CUDA)**: each level transition (`srcLod → srcLod+1`)
  is one `dispatchThreadgroups` / `cuLaunchKernel` call. Grid size is
  `(dst_w, dst_h)` at the destination mip. Same command buffer / same
  stream as the effect kernel, so no synchronization overhead beyond
  the one dependency edge between the last mip pass and the first
  effect dispatch.
- **CPU**: each level transition runs through `render_cpu_direct`,
  which partitions the destination mip across the prgpu render pool
  (bounded `num_cpus - 2` rayon workers). Same path the bench harness
  uses, no AE `iterate_with` overhead.

Because the dispatcher routes on `config.device_handle`, the same
`generate_mips(&config)` call works from GPU dispatch paths,
from Premiere's CPU-failover path, and from the `prgpu::bench` harness.

---

## Minimal CPU example

Straight from the prgpu unit test (`prgpu/src/kernels/mip.rs::tests`),
this is the complete CPU-side recipe with no AE plumbing:

```rust
use prgpu::cpu::buffer::get_or_create_with_mips;
use prgpu::kernels::mip::generate_mips;
use prgpu::types::Configuration;

const W: u32 = 32;
const H: u32 = 32;
const BPP: u32 = 4;         // Bgra8
const LEVELS: u32 = 3;

// Step 1 — allocate.
let buf = get_or_create_with_mips(W, H, BPP, LEVELS, /*tag=*/0xBEEF);

// Step 2 — fill level 0 with your source pixels.
unsafe {
    std::ptr::copy_nonoverlapping(
        source_rgba.as_ptr(),
        buf.buf.raw as *mut u8,
        (W * H * BPP) as usize,
    );
}

// Step 3 — wrap in a Configuration + generate.
let mut config = Configuration::cpu(
    buf.buf.raw,          // in_data
    buf.buf.raw,          // out_data (unused by mip_downsample)
    W as i32,             // in_pitch_px
    W as i32,             // out_pitch_px
    W, H, BPP,
    1,                    // BGRA pixel layout
);
config.outgoing_data = Some(buf.buf.raw);
config.outgoing_width = W;
config.outgoing_height = H;
config.outgoing_pitch_px = W as i32;
config.outgoing_mip_levels = LEVELS;

unsafe { generate_mips(&config).expect("mip gen failed"); }

// buf.buf.raw now contains lod 0 .. lod 2, tightly packed.
// Read mipOffsetBytes[i] from `make_outgoing_desc(&config)` or
// `fill_mip_desc(&mut desc, W, H, W, BPP, LEVELS)` to address each
// level. The effect kernel does this automatically through
// `TextureView::Load(px, lod)` / `SampleLinear(uv, lod)`.
```

---

## Minimal Metal example (GPU)

Host side inside an effect's `gpu.rs::render`:

```rust
use prgpu::gpu::backends::metal::buffer as metal_buf;
use prgpu::kernels::mip::generate_mips;
use prgpu::{Configuration, DeviceHandleInit};

// Start with the Configuration prgpu gave you (wraps Premiere's
// outgoing + dest).
let mut config = unsafe { Configuration::effect(&props, out_frame)? };
config.outgoing_mip_levels = 4;

// Step 1 — allocate a Metal buffer sized for the mip chain. Keyed on
// `MIP_TAG` so successive frames at the same dims hit the cache.
const MIP_TAG: u32 = 0xD1_D0_11_00;
let mip_buf = unsafe {
    metal_buf::get_or_create_with_mips(
        DeviceHandleInit::FromPtr(config.device_handle),
        config.outgoing_width,
        config.outgoing_height,
        config.bytes_per_pixel,
        config.outgoing_mip_levels,
        MIP_TAG,
    )
};

// Step 2 — blit Premiere's outgoing into level 0 of the mip buffer.
// (Phase 3 of the radialblur perf plan will wrap this in a helper;
// for now effects roll their own blit encoder — see MIP_BLIT_PATTERN
// at the bottom of this file.)
unsafe { blit_buffer_to_buffer(
    config.command_queue_handle as *mut _,
    config.outgoing_data.unwrap() as *mut _,   // src MTLBuffer
    mip_buf.buf.raw as *mut _,                 // dst MTLBuffer
    0,                                         // src offset
    0,                                         // dst offset = level 0
    (config.outgoing_width * config.outgoing_height * config.bytes_per_pixel) as usize,
)?; }

// Swap outgoing on the config so everything downstream reads the
// mip-backed buffer instead.
config.outgoing_data = Some(mip_buf.buf.raw);

// Step 3 — fill lods 1..N-1 and dispatch the effect. Both go through
// the same command queue, so Metal orders them automatically.
unsafe { generate_mips(&config)?; }
unsafe { pyramid_blur(&config, user_params)?; }
```

The `outgoing_width` / `outgoing_height` / `outgoing_pitch_px` stay at
level-0 dims — the dispatcher's `make_outgoing_desc(&config)` call
derives the full chain from those + `outgoing_mip_levels` via
`fill_mip_desc`. The shader gets the populated `frame.outDesc.mip*`
fields for free.

---

## Tag hygiene

Every allocator takes a `tag: u32` that participates in the cache key.
Use a unique compile-time constant per logical buffer role so two
effects (or two passes in the same effect) don't stomp on each other.
Convention: upper half = effect namespace, lower half = role.

```rust
const RADIALBLUR_MIP_TAG: u32 = 0x4842_0001;   // "HB" for Radial Blur + role 1
const GLOW_MIP_TAG: u32       = 0x474C_0001;   // "GL" for Glow + role 1
```

The allocator is 12-slot LRU, so reusing tags across frames is safe as
long as the `(w, h, bpp, mip_levels, tag)` tuple matches — the cached
buffer is returned verbatim.

---

## Debugging

- Assert `rust_texture_desc_size_matches_slang_layout` is green in
  `cargo test -p prgpu`. If it fails, the Rust `TextureDesc` grew or
  shrank without a matching vekl edit and every mip access is reading
  random bytes.
- If the mip chain looks black past level 0, check that
  `config.outgoing_data` was set to `mip_buf.buf.raw` **before** the
  `generate_mips` call — the most common mistake is leaving it
  pointing at Premiere's original outgoing (which has no mip bytes
  allocated past level 0).
- Metal console messages about `setBuffer:offset:atIndex:` with a
  zero-sized buffer mean the allocator returned a null ptr — check
  for the `[Metal] ABORT: refusing absurd buffer allocation...` log
  earlier in the run. That guard fires when `TextureDesc` byte size
  desyncs with what the host allocator thinks it should be.

---

## MIP_BLIT_PATTERN (Metal)

Representative of what the Phase 3 helper will encapsulate. Not
production-ready; use as a template.

```rust
// SAFETY: src / dst must be valid MTLBuffers of at least `size_bytes`,
// command_queue must belong to the same MTLDevice that owns both.
unsafe fn blit_buffer_to_buffer(
    command_queue: *mut objc::runtime::Object,
    src: *mut objc::runtime::Object,
    dst: *mut objc::runtime::Object,
    src_offset: usize,
    dst_offset: usize,
    size_bytes: usize,
) -> Result<(), &'static str> {
    use objc::{msg_send, sel, sel_impl};
    let cmd: *mut objc::runtime::Object = msg_send![command_queue, commandBuffer];
    if cmd.is_null() { return Err("commandBuffer"); }
    let enc: *mut objc::runtime::Object = msg_send![cmd, blitCommandEncoder];
    if enc.is_null() { return Err("blitCommandEncoder"); }
    let _: () = msg_send![enc,
        copyFromBuffer: src sourceOffset: src_offset
        toBuffer: dst destinationOffset: dst_offset
        size: size_bytes];
    let _: () = msg_send![enc, endEncoding];
    let _: () = msg_send![cmd, commit];
    // Let the caller chain its own dispatches off the same queue;
    // Metal orders them by submission order automatically.
    Ok(())
}
```
