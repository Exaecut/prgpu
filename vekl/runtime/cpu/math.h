#pragma once

#include "../../core/config.h"
#include <algorithm>
#include <cmath>

using uint = unsigned int;

namespace vekl {
    using uint = ::uint;

    inline float saturate(float x) {
        return std::max(0.0f, std::min(1.0f, x));
    }
    
    inline float rsqrt(float x) {
        return 1.0f / std::sqrt(x);
    }

    inline float abs(float x) { return std::fabs(x); }
    inline int abs(int x) { return std::abs(x); }

    inline float min(float a, float b) { return std::fmin(a, b); }
    inline uint min(uint a, uint b) { return std::min(a, b); }
    inline int min(int a, int b) { return std::min(a, b); }

    inline float max(float a, float b) { return std::fmax(a, b); }
    inline uint max(uint a, uint b) { return std::max(a, b); }
    inline int max(int a, int b) { return std::max(a, b); }
}