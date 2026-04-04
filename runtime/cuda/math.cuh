#pragma once

#include "vector.cuh"
#include <math_functions.h>

using uint = unsigned int;

namespace vekl {
    using uint = ::uint;

    __device__ inline float saturate(float x) { return __saturatef(x); }
    __device__ inline float rsqrt(float x) { return __frsqrt_rn(x); }

    inline __device__ float abs(float x) { return fabsf(x); }
    inline __device__ int abs(int x) { return ::abs(x); }

    inline __device__ float min(float a, float b) { return fminf(a, b); }
    inline __device__ int min(int a, int b) { return (a < b) ? a : b; }
    inline __device__ uint min(uint a, uint b) { return (a < b) ? a : b; }

    inline __device__ float max(float a, float b) { return fmaxf(a, b); }
    inline __device__ int max(int a, int b) { return (a > b) ? a : b; }
    inline __device__ uint max(uint a, uint b) { return (a > b) ? a : b; }
}