# GPU Frame Extension (xframe)

Effects that sample beyond the input frame boundary (e.g. blur, glow, radial blur)
need an "extended frame" — an output buffer larger than the input to accommodate
pixels that would otherwise be clipped at the edges.

## CPU Path (After Effects SmartFX)

AE's `SmartPreRender` / `SmartRender` API handles xframe natively:

1. In `SmartPreRender`, expand the result rect by the xframe amount on all sides,
   then call `extra.union_result_rect()` and `extra.union_max_result_rect()`.
2. AE allocates a larger output buffer automatically.
3. In `SmartRender`, the output layer is larger than the input. Use
   `render_cpu_direct` instead of `render_cpu` when the buffers differ in size,
   since AE's `iterate_with` requires same-sized in/out layers.

The shader compensates for the offset by subtracting `xframe / src_size` from UV
coordinates, effectively centering the original input within the expanded output.

## GPU Path (Premiere Pro)

Premiere's GPU filter API does **not** automatically allocate expanded frames.
The `out_frame` passed to `GpuFilter::render()` always matches the source dimensions.

### Solution: `GPUDeviceSuite::create_gpu_ppix()`

The Premiere SDK provides `create_gpu_ppix()` which allocates a new GPU pixel buffer
with custom dimensions. This allows the effect to:

1. Compute the required xframe based on maximum blur sample displacement (e.g. `RadialblurParams::compute_xframe()`).
2. Allocate a larger output frame via `filter.gpu_device_suite.create_gpu_ppix()`.
3. Override the `Configuration` with the new buffer's data pointer, pitch, and
   expanded dimensions.
4. Dispatch the kernel into the larger buffer.
5. Replace `*out_frame` with the new PPix handle — Premiere accepts this per the
   SDK documentation: *"it is allowable for the effect to allocate and return a
   different sized outFrame."*

### API Signature

```c
PrGPUDeviceSuite::create_gpu_ppix(
    gpu_index: u32,
    pixel_format: PrPixelFormat,
    width: i32,
    height: i32,
    par_num: i32,
    par_den: i32,
    field_type: prFieldType,
) -> Result<PPixHand, Error>
```

### Cleanup

The allocated PPix is owned by Premiere after `*out_frame` replacement — the host
is responsible for disposing it. Do **not** call `PPixSuite::dispose()` on the
new frame yourself.

### Reference Implementation

See `radialblur/src/gpu.rs` for a complete example of GPU xframe allocation
and dispatch.

## UV Mapping in the Shader

Both CPU and GPU paths use the same VEKL shader, which expects:

- `params.xframe` — the xframe amount in pixels (scalar, same for X and Y), computed from maximum blur displacement
- `params.src_width` / `params.src_height` — the original (unexpanded) dimensions

The shader maps output pixel coordinates back to input UV space:

```hlsl
float2 xframe_offset = params.xframe / src_size_f;
float2 uv = tex_coord(gid, src_size) - xframe_offset;
```

This centers the original image within the expanded output, with `xframe` pixels
of padding on each side.

## CPU vs GPU Xframe Differences

| Aspect | CPU (AE SmartFX) | GPU (Premiere) |
|--------|-------------------|----------------|
| Buffer allocation | Automatic (AE) | Manual (`create_gpu_ppix`) |
| Output frame replacement | N/A | `*out_frame = new_ppix` |
| Dispatch method | `render_cpu_direct` | Standard GPU dispatch |
| Cleanup | AE handles | Premiere handles (after replacement) |
| Symmetric expansion | `2 × xframe` on all sides | `2 × xframe` on all sides |
