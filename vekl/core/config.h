#pragma once

#define VEKL_VERSION_MAJOR 0
#define VEKL_VERSION_MINOR 1
#define VEKL_VERSION_PATCH 0

#if defined(_MSC_VER)
  #define VEKL_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define VEKL_FORCE_INLINE inline __attribute__((always_inline))
#else
  #define VEKL_FORCE_INLINE inline
#endif

#define VEKL_UNUSED(x) (void)(x)
#define VEKL_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))