#pragma once

#include "core/config.h"
#include "core/traits.h"

#if defined(VEKL_FORCE_CUDA)
  #define VEKL_USE_CUDA 1
#elif defined(VEKL_FORCE_METAL)
  #define VEKL_USE_METAL 1
#elif defined(VEKL_FORCE_CPU)
  #define VEKL_USE_CPU 1
#elif defined(__CUDACC__)
  #define VEKL_USE_CUDA 1
#elif defined(__METAL_VERSION__) || defined(VEKL_METAL)
  #define VEKL_USE_METAL 1
#else
  #define VEKL_USE_CPU 1
#endif

#if defined(VEKL_USE_CUDA)
    #include "runtime/cuda/defines.cuh"
    #include "runtime/cuda/vector.cuh"
    #include "runtime/cuda/math.cuh"
#elif defined(VEKL_USE_METAL)
    #include <metal_stdlib>
    using namespace metal;
    
    #include "runtime/metal/defines.metal"
    #include "runtime/metal/vector.metal"
    #include "runtime/metal/math.metal"
#else
    #include "runtime/cpu/defines.h"
    #include "runtime/cpu/vector.h"
    #include "runtime/cpu/math.h"
#endif

// Shared Core Utilities
#include "core/math.h"

using namespace vekl;