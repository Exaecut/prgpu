#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
    #define VEKL_HD __host__ __device__ __forceinline__
#else
    #define VEKL_HD inline
#endif

using uint = unsigned int;
using ushort = unsigned short;

// minimal void_t
template <typename...> struct vekl_void { using type = void; };
template <typename... Ts> using vekl_void_t = typename vekl_void<Ts...>::type;

#define VEKL_VEC1_CTORS(T, name, make1) \
    VEKL_HD T##1 name(T x) { return make1(x); } \
    template <typename V, typename = vekl_void_t<decltype(std::declval<V>().x)>> \
    VEKL_HD T##1 name(const V& v) { return make1(static_cast<T>(v.x)); }

#define VEKL_VEC2_CTORS(T, name, make2) \
    VEKL_HD T##2 name(T v) { return make2(v, v); } \
    VEKL_HD T##2 name(T x, T y) { return make2(x, y); } \
    template <typename V, typename = vekl_void_t<decltype(std::declval<V>().x), decltype(std::declval<V>().y)>> \
    VEKL_HD T##2 name(const V& v) { return make2(static_cast<T>(v.x), static_cast<T>(v.y)); }

#define VEKL_VEC3_CTORS(T, name, make3) \
    VEKL_HD T##3 name(T v) { return make3(v, v, v); } \
    VEKL_HD T##3 name(T x, T y, T z) { return make3(x, y, z); } \
    VEKL_HD T##3 name(T##2 xy, T z) { return make3(xy.x, xy.y, z); } \
    VEKL_HD T##3 name(T x, T##2 yz) { return make3(x, yz.x, yz.y); } \
    template <typename V, typename = vekl_void_t<decltype(std::declval<V>().x), decltype(std::declval<V>().y), decltype(std::declval<V>().z)>> \
    VEKL_HD T##3 name(const V& v) { return make3(static_cast<T>(v.x), static_cast<T>(v.y), static_cast<T>(v.z)); }

#define VEKL_VEC4_CTORS(T, name, make4) \
    VEKL_HD T##4 name(T v) { return make4(v, v, v, v); } \
    VEKL_HD T##4 name(T x, T y, T z, T w) { return make4(x, y, z, w); } \
    VEKL_HD T##4 name(T##2 xy, T##2 zw) { return make4(xy.x, xy.y, zw.x, zw.y); } \
    VEKL_HD T##4 name(T##2 xy, T z, T w) { return make4(xy.x, xy.y, z, w); } \
    VEKL_HD T##4 name(T x, T##2 yz, T w) { return make4(x, yz.x, yz.y, w); } \
    VEKL_HD T##4 name(T x, T y, T##2 zw) { return make4(x, y, zw.x, zw.y); } \
    VEKL_HD T##4 name(T##3 xyz, T w) { return make4(xyz.x, xyz.y, xyz.z, w); } \
    VEKL_HD T##4 name(T x, T##3 yzw) { return make4(x, yzw.x, yzw.y, yzw.z); } \
    template <typename V, typename = vekl_void_t<decltype(std::declval<V>().x), decltype(std::declval<V>().y), decltype(std::declval<V>().z), decltype(std::declval<V>().w)>> \
    VEKL_HD T##4 name(const V& v) { return make4(static_cast<T>(v.x), static_cast<T>(v.y), static_cast<T>(v.z), static_cast<T>(v.w)); }

#define VEKL_DEFINE_ALL(T, make1, make2, make3, make4) \
    VEKL_VEC1_CTORS(T, vekl_make_##T##1, make1) \
    VEKL_VEC2_CTORS(T, vekl_make_##T##2, make2) \
    VEKL_VEC3_CTORS(T, vekl_make_##T##3, make3) \
    VEKL_VEC4_CTORS(T, vekl_make_##T##4, make4)

// helpers in global namespace
VEKL_DEFINE_ALL(char,      make_char1,      make_char2,      make_char3,      make_char4)
VEKL_DEFINE_ALL(short,     make_short1,     make_short2,     make_short3,     make_short4)
VEKL_DEFINE_ALL(ushort,    make_ushort1,    make_ushort2,    make_ushort3,    make_ushort4)
VEKL_DEFINE_ALL(int,       make_int1,       make_int2,       make_int3,       make_int4)
VEKL_DEFINE_ALL(uint,      make_uint1,      make_uint2,      make_uint3,      make_uint4)
VEKL_DEFINE_ALL(long,      make_long1,      make_long2,      make_long3,      make_long4)
VEKL_DEFINE_ALL(float,     make_float1,     make_float2,     make_float3,     make_float4)
VEKL_DEFINE_ALL(double,    make_double1,    make_double2,    make_double3,    make_double4)

// macro bridge to preserve `float2(...)` syntax
#define char1(...)      vekl_make_char1(__VA_ARGS__)
#define char2(...)      vekl_make_char2(__VA_ARGS__)
#define char3(...)      vekl_make_char3(__VA_ARGS__)
#define char4(...)      vekl_make_char4(__VA_ARGS__)

#define uchar1(...)     vekl_make_uchar1(__VA_ARGS__)
#define uchar2(...)     vekl_make_uchar2(__VA_ARGS__)
#define uchar3(...)     vekl_make_uchar3(__VA_ARGS__)
#define uchar4(...)     vekl_make_uchar4(__VA_ARGS__)

#define short1(...)     vekl_make_short1(__VA_ARGS__)
#define short2(...)     vekl_make_short2(__VA_ARGS__)
#define short3(...)     vekl_make_short3(__VA_ARGS__)
#define short4(...)     vekl_make_short4(__VA_ARGS__)

#define ushort1(...)    vekl_make_ushort1(__VA_ARGS__)
#define ushort2(...)    vekl_make_ushort2(__VA_ARGS__)
#define ushort3(...)    vekl_make_ushort3(__VA_ARGS__)
#define ushort4(...)    vekl_make_ushort4(__VA_ARGS__)

#define int1(...)       vekl_make_int1(__VA_ARGS__)
#define int2(...)       vekl_make_int2(__VA_ARGS__)
#define int3(...)       vekl_make_int3(__VA_ARGS__)
#define int4(...)       vekl_make_int4(__VA_ARGS__)

#define uint1(...)      vekl_make_uint1(__VA_ARGS__)
#define uint2(...)      vekl_make_uint2(__VA_ARGS__)
#define uint3(...)      vekl_make_uint3(__VA_ARGS__)
#define uint4(...)      vekl_make_uint4(__VA_ARGS__)

#define long1(...)      vekl_make_long1(__VA_ARGS__)
#define long2(...)      vekl_make_long2(__VA_ARGS__)
#define long3(...)      vekl_make_long3(__VA_ARGS__)
#define long4(...)      vekl_make_long4(__VA_ARGS__)

#define ulong1(...)     vekl_make_ulong1(__VA_ARGS__)
#define ulong2(...)     vekl_make_ulong2(__VA_ARGS__)
#define ulong3(...)     vekl_make_ulong3(__VA_ARGS__)
#define ulong4(...)     vekl_make_ulong4(__VA_ARGS__)

#define longlong1(...)  vekl_make_longlong1(__VA_ARGS__)
#define longlong2(...)  vekl_make_longlong2(__VA_ARGS__)
#define longlong3(...)  vekl_make_longlong3(__VA_ARGS__)
#define longlong4(...)  vekl_make_longlong4(__VA_ARGS__)

#define ulonglong1(...) vekl_make_ulonglong1(__VA_ARGS__)
#define ulonglong2(...) vekl_make_ulonglong2(__VA_ARGS__)
#define ulonglong3(...) vekl_make_ulonglong3(__VA_ARGS__)
#define ulonglong4(...) vekl_make_ulonglong4(__VA_ARGS__)

#define float1(...)     vekl_make_float1(__VA_ARGS__)
#define float2(...)     vekl_make_float2(__VA_ARGS__)
#define float3(...)     vekl_make_float3(__VA_ARGS__)
#define float4(...)     vekl_make_float4(__VA_ARGS__)

#define double1(...)    vekl_make_double1(__VA_ARGS__)
#define double2(...)    vekl_make_double2(__VA_ARGS__)
#define double3(...)    vekl_make_double3(__VA_ARGS__)
#define double4(...)    vekl_make_double4(__VA_ARGS__)

#define VEKL_VEC1_OPS(T, make1) \
    VEKL_HD T##1 operator+(T##1 a, T##1 b) { return make1(a.x + b.x); } \
    VEKL_HD T##1 operator+(T##1 a, T b)   { return make1(a.x + b); } \
    VEKL_HD T##1 operator+(T a, T##1 b)   { return make1(a + b.x); } \
    VEKL_HD T##1 operator-(T##1 a, T##1 b) { return make1(a.x - b.x); } \
    VEKL_HD T##1 operator-(T##1 a, T b)   { return make1(a.x - b); } \
    VEKL_HD T##1 operator-(T a, T##1 b)   { return make1(a - b.x); } \
    VEKL_HD T##1 operator*(T##1 a, T##1 b) { return make1(a.x * b.x); } \
    VEKL_HD T##1 operator*(T##1 a, T b)   { return make1(a.x * b); } \
    VEKL_HD T##1 operator*(T a, T##1 b)   { return make1(a * b.x); } \
    VEKL_HD T##1 operator/(T##1 a, T##1 b) { return make1(a.x / b.x); } \
    VEKL_HD T##1 operator/(T##1 a, T b)   { return make1(a.x / b); } \
    VEKL_HD T##1 operator/(T a, T##1 b)   { return make1(a / b.x); } \
    VEKL_HD T##1 operator-(T##1 a)        { return make1(-a.x); } \
    VEKL_HD T##1 operator+(T##1 a)        { return a; } \
    VEKL_HD T##1& operator+=(T##1& a, T##1 b) { a.x += b.x; return a; } \
    VEKL_HD T##1& operator+=(T##1& a, T b)    { a.x += b; return a; } \
    VEKL_HD T##1& operator-=(T##1& a, T##1 b) { a.x -= b.x; return a; } \
    VEKL_HD T##1& operator-=(T##1& a, T b)    { a.x -= b; return a; } \
    VEKL_HD T##1& operator*=(T##1& a, T##1 b) { a.x *= b.x; return a; } \
    VEKL_HD T##1& operator*=(T##1& a, T b)    { a.x *= b; return a; } \
    VEKL_HD T##1& operator/=(T##1& a, T##1 b) { a.x /= b.x; return a; } \
    VEKL_HD T##1& operator/=(T##1& a, T b)    { a.x /= b; return a; }

#define VEKL_VEC2_OPS(T, make2) \
    VEKL_HD T##2 operator+(T##2 a, T##2 b) { return make2(a.x + b.x, a.y + b.y); } \
    VEKL_HD T##2 operator+(T##2 a, T b)   { return make2(a.x + b, a.y + b); } \
    VEKL_HD T##2 operator+(T a, T##2 b)   { return make2(a + b.x, a + b.y); } \
    VEKL_HD T##2 operator-(T##2 a, T##2 b) { return make2(a.x - b.x, a.y - b.y); } \
    VEKL_HD T##2 operator-(T##2 a, T b)   { return make2(a.x - b, a.y - b); } \
    VEKL_HD T##2 operator-(T a, T##2 b)   { return make2(a - b.x, a - b.y); } \
    VEKL_HD T##2 operator*(T##2 a, T##2 b) { return make2(a.x * b.x, a.y * b.y); } \
    VEKL_HD T##2 operator*(T##2 a, T b)   { return make2(a.x * b, a.y * b); } \
    VEKL_HD T##2 operator*(T a, T##2 b)   { return make2(a * b.x, a * b.y); } \
    VEKL_HD T##2 operator/(T##2 a, T##2 b) { return make2(a.x / b.x, a.y / b.y); } \
    VEKL_HD T##2 operator/(T##2 a, T b)   { return make2(a.x / b, a.y / b); } \
    VEKL_HD T##2 operator/(T a, T##2 b)   { return make2(a / b.x, a / b.y); } \
    VEKL_HD T##2 operator-(T##2 a)        { return make2(-a.x, -a.y); } \
    VEKL_HD T##2 operator+(T##2 a)        { return a; } \
    VEKL_HD T##2& operator+=(T##2& a, T##2 b) { a.x += b.x; a.y += b.y; return a; } \
    VEKL_HD T##2& operator+=(T##2& a, T b)    { a.x += b; a.y += b; return a; } \
    VEKL_HD T##2& operator-=(T##2& a, T##2 b) { a.x -= b.x; a.y -= b.y; return a; } \
    VEKL_HD T##2& operator-=(T##2& a, T b)    { a.x -= b; a.y -= b; return a; } \
    VEKL_HD T##2& operator*=(T##2& a, T##2 b) { a.x *= b.x; a.y *= b.y; return a; } \
    VEKL_HD T##2& operator*=(T##2& a, T b)    { a.x *= b; a.y *= b; return a; } \
    VEKL_HD T##2& operator/=(T##2& a, T##2 b) { a.x /= b.x; a.y /= b.y; return a; } \
    VEKL_HD T##2& operator/=(T##2& a, T b)    { a.x /= b; a.y /= b; return a; }

#define VEKL_VEC3_OPS(T, make3) \
    VEKL_HD T##3 operator+(T##3 a, T##3 b) { return make3(a.x + b.x, a.y + b.y, a.z + b.z); } \
    VEKL_HD T##3 operator+(T##3 a, T b)   { return make3(a.x + b, a.y + b, a.z + b); } \
    VEKL_HD T##3 operator+(T a, T##3 b)   { return make3(a + b.x, a + b.y, a + b.z); } \
    VEKL_HD T##3 operator-(T##3 a, T##3 b) { return make3(a.x - b.x, a.y - b.y, a.z - b.z); } \
    VEKL_HD T##3 operator-(T##3 a, T b)   { return make3(a.x - b, a.y - b, a.z - b); } \
    VEKL_HD T##3 operator-(T a, T##3 b)   { return make3(a - b.x, a - b.y, a - b.z); } \
    VEKL_HD T##3 operator*(T##3 a, T##3 b) { return make3(a.x * b.x, a.y * b.y, a.z * b.z); } \
    VEKL_HD T##3 operator*(T##3 a, T b)   { return make3(a.x * b, a.y * b, a.z * b); } \
    VEKL_HD T##3 operator*(T a, T##3 b)   { return make3(a * b.x, a * b.y, a * b.z); } \
    VEKL_HD T##3 operator/(T##3 a, T##3 b) { return make3(a.x / b.x, a.y / b.y, a.z / b.z); } \
    VEKL_HD T##3 operator/(T##3 a, T b)   { return make3(a.x / b, a.y / b, a.z / b); } \
    VEKL_HD T##3 operator/(T a, T##3 b)   { return make3(a / b.x, a / b.y, a / b.z); } \
    VEKL_HD T##3 operator-(T##3 a)        { return make3(-a.x, -a.y, -a.z); } \
    VEKL_HD T##3 operator+(T##3 a)        { return a; } \
    VEKL_HD T##3& operator+=(T##3& a, T##3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; } \
    VEKL_HD T##3& operator+=(T##3& a, T b)    { a.x += b; a.y += b; a.z += b; return a; } \
    VEKL_HD T##3& operator-=(T##3& a, T##3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; } \
    VEKL_HD T##3& operator-=(T##3& a, T b)    { a.x -= b; a.y -= b; a.z -= b; return a; } \
    VEKL_HD T##3& operator*=(T##3& a, T##3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; } \
    VEKL_HD T##3& operator*=(T##3& a, T b)    { a.x *= b; a.y *= b; a.z *= b; return a; } \
    VEKL_HD T##3& operator/=(T##3& a, T##3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; } \
    VEKL_HD T##3& operator/=(T##3& a, T b)    { a.x /= b; a.y /= b; a.z /= b; return a; }

#define VEKL_VEC4_OPS(T, make4) \
    VEKL_HD T##4 operator+(T##4 a, T##4 b) { return make4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); } \
    VEKL_HD T##4 operator+(T##4 a, T b)   { return make4(a.x + b, a.y + b, a.z + b, a.w + b); } \
    VEKL_HD T##4 operator+(T a, T##4 b)   { return make4(a + b.x, a + b.y, a + b.z, a + b.w); } \
    VEKL_HD T##4 operator-(T##4 a, T##4 b) { return make4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); } \
    VEKL_HD T##4 operator-(T##4 a, T b)   { return make4(a.x - b, a.y - b, a.z - b, a.w - b); } \
    VEKL_HD T##4 operator-(T a, T##4 b)   { return make4(a - b.x, a - b.y, a - b.z, a - b.w); } \
    VEKL_HD T##4 operator*(T##4 a, T##4 b) { return make4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); } \
    VEKL_HD T##4 operator*(T##4 a, T b)   { return make4(a.x * b, a.y * b, a.z * b, a.w * b); } \
    VEKL_HD T##4 operator*(T a, T##4 b)   { return make4(a * b.x, a * b.y, a * b.z, a * b.w); } \
    VEKL_HD T##4 operator/(T##4 a, T##4 b) { return make4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); } \
    VEKL_HD T##4 operator/(T##4 a, T b)   { return make4(a.x / b, a.y / b, a.z / b, a.w / b); } \
    VEKL_HD T##4 operator/(T a, T##4 b)   { return make4(a / b.x, a / b.y, a / b.z, a / b.w); } \
    VEKL_HD T##4 operator-(T##4 a)        { return make4(-a.x, -a.y, -a.z, -a.w); } \
    VEKL_HD T##4 operator+(T##4 a)        { return a; } \
    VEKL_HD T##4& operator+=(T##4& a, T##4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; } \
    VEKL_HD T##4& operator+=(T##4& a, T b)    { a.x += b; a.y += b; a.z += b; a.w += b; return a; } \
    VEKL_HD T##4& operator-=(T##4& a, T##4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; } \
    VEKL_HD T##4& operator-=(T##4& a, T b)    { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; } \
    VEKL_HD T##4& operator*=(T##4& a, T##4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; } \
    VEKL_HD T##4& operator*=(T##4& a, T b)    { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; } \
    VEKL_HD T##4& operator/=(T##4& a, T##4 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; } \
    VEKL_HD T##4& operator/=(T##4& a, T b)    { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }

#define VEKL_DEFINE_OPS(T, make1, make2, make3, make4) \
    VEKL_VEC1_OPS(T, make1) \
    VEKL_VEC2_OPS(T, make2) \
    VEKL_VEC3_OPS(T, make3) \
    VEKL_VEC4_OPS(T, make4)

// Apply to all native CUDA vector base types
VEKL_DEFINE_OPS(char,      make_char1,      make_char2,      make_char3,      make_char4)
VEKL_DEFINE_OPS(short,     make_short1,     make_short2,     make_short3,     make_short4)
VEKL_DEFINE_OPS(ushort,    make_ushort1,    make_ushort2,    make_ushort3,    make_ushort4)
VEKL_DEFINE_OPS(int,       make_int1,       make_int2,       make_int3,       make_int4)
VEKL_DEFINE_OPS(uint,      make_uint1,      make_uint2,      make_uint3,      make_uint4)
VEKL_DEFINE_OPS(long,      make_long1,      make_long2,      make_long3,      make_long4)
VEKL_DEFINE_OPS(float,     make_float1,     make_float2,     make_float3,     make_float4)
VEKL_DEFINE_OPS(double,    make_double1,    make_double2,    make_double3,    make_double4)

#undef VEKL_DEFINE_OPS
#undef VEKL_VEC4_OPS
#undef VEKL_VEC3_OPS
#undef VEKL_VEC2_OPS
#undef VEKL_VEC1_OPS

#undef VEKL_DEFINE_ALL
#undef VEKL_VEC1_CTORS
#undef VEKL_VEC2_CTORS
#undef VEKL_VEC3_CTORS
#undef VEKL_VEC4_CTORS
#undef VEKL_HD