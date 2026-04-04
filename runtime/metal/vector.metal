#pragma once

#include <metal_stdlib>
#include "../../core/config.h"

namespace vekl {
    using uint = metal::uint;
    using int_ = metal::int_;
    
    using float2 = metal::float2;
    using float3 = metal::float3;
    using float4 = metal::float4;
    
    using int2 = metal::int2;
    using int3 = metal::int3;
    using int4 = metal::int4;
    
    using uint2 = metal::uint2;
    using uint3 = metal::uint3;
    using uint4 = metal::uint4;
    
    using bool2 = metal::bool2;
    
    template<typename T>
    inline float2 make_float2(T x, T y) { return float2(x, y); }
    
    template<typename T>
    inline uint2 make_uint2(T x, T y) { return uint2(x, y); }
}
