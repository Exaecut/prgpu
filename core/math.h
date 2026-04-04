#pragma once

#include "config.h"

namespace vekl {
    template<typename T>
    inline T abs(T x) {
        return x < T(0) ? -x : x;
    }

    template<typename T>
    inline T min(T a, T b) {
        return a < b ? a : b;
    }

    template<typename T>
    inline T max(T a, T b) {
        return a > b ? a : b;
    }
}