# VEKL (Video Effects Kernel Language)

VEKL is an open-source, cross-platform compute kernel foundation designed for developing high-performance video effects and transitions for Adobe plugins and similar pipelines. It abstracts the differences between CUDA, Metal, and CPU backend execution into a single, unified, Metal-like DSL.

## Supported Backends

1. **CUDA** (NVCC)
2. **Metal** (MSL)
3. **CPU** (C++17 Fallback)

## Quick Start

1. Copy the `vekl` folder into your project's include path.
2. Include the master header:

   ```cpp
      #include "vekl/vekl.h"
   ```

3. Write your kernel once:

   ```cpp
   struct Params {
      float threshold;
      float4 tint;
   };

   // Unified kernel signature
   kernel void apply_tint_filter(
      param_ro(float4, src, 0),    // Read-only buffer
      param_wo(float4, dst, 1),    // Write-only buffer
      param_cbuf(Params, p, 2),    // Constant buffer / Uniforms
      thread_pos_param(gid)        // Thread position (Metal specific arg, CUDA ignored)
   ) {
      // Initialize thread position for CUDA/CPU (no-op on Metal)
      thread_pos_init(gid);

      // Guard bounds (assuming 1920x1080 linear buffer for example)
      vekl::uint idx = gid.y * 1920 + gid.x;
      
      // Native vector types and swizzles
      float4 pixel = src[idx];
      
      // Unified math (vekl::min, vekl::saturate, etc.)
      float lum = vekl::dot(pixel.rgb, float3(0.2126, 0.7152, 0.0722));
      
      if (lum > p.threshold) {
         pixel = vekl::mix(pixel, p.tint, 0.5f);
      }
      
      dst[idx] = pixel;
   }
   ```

## Structure

- vekl/vekl.h: Main entry point.
- vekl/core/: Platform detection and shared math traits.
- vekl/runtime/: Backend-specific implementations (CUDA/Metal/CPU).

## Contributing

Contributions are welcome. Please ensure any new helpers are added to `vekl` namespace and have implementations for all backends.
