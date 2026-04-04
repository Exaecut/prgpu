#pragma once

#define VEKL_HD inline

namespace vekl {

using uint = unsigned int;

template<bool B, typename T = void>
struct enable_if {};
template<typename T>
struct enable_if<true, T> { using type = T; };
template<bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

template<typename A, typename B>
struct is_same { static constexpr bool value = false; };
template<typename A>
struct is_same<A, A> { static constexpr bool value = true; };

template<typename T> struct is_scalar { static constexpr bool value = false; };
template<> struct is_scalar<bool> { static constexpr bool value = true; };
template<> struct is_scalar<int> { static constexpr bool value = true; };
template<> struct is_scalar<uint> { static constexpr bool value = true; };
template<> struct is_scalar<float> { static constexpr bool value = true; };
template<> struct is_scalar<double> { static constexpr bool value = true; };

template<typename T> struct is_float { static constexpr bool value = false; };
template<> struct is_float<float> { static constexpr bool value = true; };
template<> struct is_float<double> { static constexpr bool value = true; };

template<typename T> struct is_int { static constexpr bool value = false; };
template<> struct is_int<int> { static constexpr bool value = true; };

template<typename T> struct is_uint { static constexpr bool value = false; };
template<> struct is_uint<uint> { static constexpr bool value = true; };

template<typename T> struct is_bool { static constexpr bool value = false; };
template<> struct is_bool<bool> { static constexpr bool value = true; };

template<bool B, typename T, typename F>
struct type_select { using type = T; };
template<typename T, typename F>
struct type_select<false, T, F> { using type = F; };

template<typename A, typename B>
struct promote_scalar {
  using type = typename type_select<
    is_float<A>::value || is_float<B>::value,
    float,
    typename type_select<
      is_int<A>::value || is_int<B>::value,
      int,
      typename type_select<
        is_uint<A>::value || is_uint<B>::value,
        uint,
        int
      >::type
    >::type
  >::type;
};

template<typename To, typename From>
VEKL_HD constexpr To cast_to(From v) {
  if constexpr (is_same<To, bool>::value) {
    return v != 0;
  } else if constexpr (is_same<From, bool>::value) {
    return v ? To(1) : To(0);
  } else {
    return static_cast<To>(v);
  }
}

template<typename T>
VEKL_HD constexpr T default_w() { return T(0); }
template<>
VEKL_HD constexpr float default_w<float>() { return 1.0f; }

template<typename V>
struct vec_traits { using base = void; static constexpr int count = 0; };
template<typename V>
struct vec_traits<const V> : vec_traits<V> {};
template<typename V>
struct vec_traits<volatile V> : vec_traits<V> {};
template<typename V>
struct vec_traits<const volatile V> : vec_traits<V> {};

template<typename T, int N>
struct vec_type;
template<typename T, int N>
using vec_type_t = typename vec_type<T, N>::type;

// examples:
// float2(uint2(1,2))
// float3(true)
// float4(float3(1.0f,2.0f,3.0f))
// float3(float2(1,2), 3)
// float3 a; uint2 b; auto r = a * b; // z pass-through

#define VEKL_VEC2_CTORS(T, name) \
  VEKL_HD name##2() = default; \
  template<typename S, typename = enable_if_t<is_scalar<S>::value>> \
  VEKL_HD constexpr name##2(S s) : x(cast_to<T>(s)), y(cast_to<T>(s)) {} \
  template<typename A, typename B, typename = enable_if_t<is_scalar<A>::value && is_scalar<B>::value>> \
  VEKL_HD constexpr name##2(A a, B b) : x(cast_to<T>(a)), y(cast_to<T>(b)) {} \
  template<typename V, typename = enable_if_t<(vec_traits<V>::count >= 2)>> \
  VEKL_HD constexpr name##2(const V& v) : x(cast_to<T>(v.x)), y(cast_to<T>(v.y)) {}

#define VEKL_VEC3_CTORS(T, name) \
  VEKL_HD name##3() = default; \
  template<typename S, typename = enable_if_t<is_scalar<S>::value>> \
  VEKL_HD constexpr name##3(S s) : x(cast_to<T>(s)), y(cast_to<T>(s)), z(cast_to<T>(s)) {} \
  template<typename A, typename B, typename C, typename = enable_if_t<is_scalar<A>::value && is_scalar<B>::value && is_scalar<C>::value>> \
  VEKL_HD constexpr name##3(A a, B b, C c) : x(cast_to<T>(a)), y(cast_to<T>(b)), z(cast_to<T>(c)) {} \
  template<typename V, typename = enable_if_t<(vec_traits<V>::count >= 3)>> \
  VEKL_HD constexpr name##3(const V& v) : x(cast_to<T>(v.x)), y(cast_to<T>(v.y)), z(cast_to<T>(v.z)) {} \
  template<typename V2, typename S, typename = enable_if_t<(vec_traits<V2>::count == 2) && is_scalar<S>::value>> \
  VEKL_HD constexpr name##3(const V2& v, S s) : x(cast_to<T>(v.x)), y(cast_to<T>(v.y)), z(cast_to<T>(s)) {} \
  template<typename S, typename V2, typename = enable_if_t<is_scalar<S>::value && (vec_traits<V2>::count == 2)>> \
  VEKL_HD constexpr name##3(S s, const V2& v) : x(cast_to<T>(s)), y(cast_to<T>(v.x)), z(cast_to<T>(v.y)) {}

#define VEKL_VEC4_CTORS(T, name) \
  VEKL_HD name##4() = default; \
  template<typename S, typename = enable_if_t<is_scalar<S>::value>> \
  VEKL_HD constexpr name##4(S s) : x(cast_to<T>(s)), y(cast_to<T>(s)), z(cast_to<T>(s)), w(cast_to<T>(s)) {} \
  template<typename A, typename B, typename C, typename D, typename = enable_if_t<is_scalar<A>::value && is_scalar<B>::value && is_scalar<C>::value && is_scalar<D>::value>> \
  VEKL_HD constexpr name##4(A a, B b, C c, D d) : x(cast_to<T>(a)), y(cast_to<T>(b)), z(cast_to<T>(c)), w(cast_to<T>(d)) {} \
  template<typename V, typename = enable_if_t<(vec_traits<V>::count >= 4)>> \
  VEKL_HD constexpr name##4(const V& v) : x(cast_to<T>(v.x)), y(cast_to<T>(v.y)), z(cast_to<T>(v.z)), w(cast_to<T>(v.w)) {} \
  template<typename V3, typename = enable_if_t<(vec_traits<V3>::count == 3)>> \
  VEKL_HD constexpr name##4(const V3& v) : x(cast_to<T>(v.x)), y(cast_to<T>(v.y)), z(cast_to<T>(v.z)), w(default_w<T>()) {} \
  template<typename V3, typename S, typename = enable_if_t<(vec_traits<V3>::count == 3) && is_scalar<S>::value>> \
  VEKL_HD constexpr name##4(const V3& v, S s) : x(cast_to<T>(v.x)), y(cast_to<T>(v.y)), z(cast_to<T>(v.z)), w(cast_to<T>(s)) {} \
  template<typename S, typename V3, typename = enable_if_t<is_scalar<S>::value && (vec_traits<V3>::count == 3)>> \
  VEKL_HD constexpr name##4(S s, const V3& v) : x(cast_to<T>(s)), y(cast_to<T>(v.x)), z(cast_to<T>(v.y)), w(cast_to<T>(v.z)) {} \
  template<typename V2a, typename V2b, typename = enable_if_t<(vec_traits<V2a>::count == 2) && (vec_traits<V2b>::count == 2)>> \
  VEKL_HD constexpr name##4(const V2a& a, const V2b& b) : x(cast_to<T>(a.x)), y(cast_to<T>(a.y)), z(cast_to<T>(b.x)), w(cast_to<T>(b.y)) {} \
  template<typename S1, typename S2, typename V2, typename = enable_if_t<is_scalar<S1>::value && is_scalar<S2>::value && (vec_traits<V2>::count == 2)>> \
  VEKL_HD constexpr name##4(S1 s1, S2 s2, const V2& v) : x(cast_to<T>(s1)), y(cast_to<T>(s2)), z(cast_to<T>(v.x)), w(cast_to<T>(v.y)) {}

#define VEKL_ALIGN2(T) alignas(sizeof(T) * 2)
#define VEKL_ALIGN3(T) alignas(sizeof(T) * 4)
#define VEKL_ALIGN4(T) alignas(sizeof(T) * 4)

#define VEKL_DEFINE_VEC2(T, name) \
struct VEKL_ALIGN2(T) name##2 { \
  using base = T; \
  union { struct { T x, y; }; struct { T r, g; }; }; \
  VEKL_VEC2_CTORS(T, name) \
};

#define VEKL_DEFINE_VEC3(T, name) \
struct VEKL_ALIGN3(T) name##3 { \
  using base = T; \
  union { struct { T x, y, z; }; struct { T r, g, b; }; }; \
  VEKL_VEC3_CTORS(T, name) \
};

#define VEKL_DEFINE_VEC4(T, name) \
struct VEKL_ALIGN4(T) name##4 { \
  using base = T; \
  union { struct { T x, y, z, w; }; struct { T r, g, b, a; }; }; \
  VEKL_VEC4_CTORS(T, name) \
  VEKL_HD constexpr name##3 rgb() const { return name##3(x, y, z); } \
};

VEKL_DEFINE_VEC2(float, float)
VEKL_DEFINE_VEC3(float, float)
VEKL_DEFINE_VEC4(float, float)

VEKL_DEFINE_VEC2(int, int)
VEKL_DEFINE_VEC3(int, int)
VEKL_DEFINE_VEC4(int, int)

VEKL_DEFINE_VEC2(uint, uint)
VEKL_DEFINE_VEC3(uint, uint)
VEKL_DEFINE_VEC4(uint, uint)

VEKL_DEFINE_VEC2(bool, bool)
VEKL_DEFINE_VEC3(bool, bool)
VEKL_DEFINE_VEC4(bool, bool)

template<> struct vec_traits<float2> { using base = float; static constexpr int count = 2; };
template<> struct vec_traits<float3> { using base = float; static constexpr int count = 3; };
template<> struct vec_traits<float4> { using base = float; static constexpr int count = 4; };

template<> struct vec_traits<int2> { using base = int; static constexpr int count = 2; };
template<> struct vec_traits<int3> { using base = int; static constexpr int count = 3; };
template<> struct vec_traits<int4> { using base = int; static constexpr int count = 4; };

template<> struct vec_traits<uint2> { using base = uint; static constexpr int count = 2; };
template<> struct vec_traits<uint3> { using base = uint; static constexpr int count = 3; };
template<> struct vec_traits<uint4> { using base = uint; static constexpr int count = 4; };

template<> struct vec_traits<bool2> { using base = bool; static constexpr int count = 2; };
template<> struct vec_traits<bool3> { using base = bool; static constexpr int count = 3; };
template<> struct vec_traits<bool4> { using base = bool; static constexpr int count = 4; };

template<> struct vec_type<float, 2> { using type = float2; };
template<> struct vec_type<float, 3> { using type = float3; };
template<> struct vec_type<float, 4> { using type = float4; };

template<> struct vec_type<int, 2> { using type = int2; };
template<> struct vec_type<int, 3> { using type = int3; };
template<> struct vec_type<int, 4> { using type = int4; };

template<> struct vec_type<uint, 2> { using type = uint2; };
template<> struct vec_type<uint, 3> { using type = uint3; };
template<> struct vec_type<uint, 4> { using type = uint4; };

template<> struct vec_type<bool, 2> { using type = bool2; };
template<> struct vec_type<bool, 3> { using type = bool3; };
template<> struct vec_type<bool, 4> { using type = bool4; };

template<typename A, typename B>
VEKL_HD constexpr float2 make_float2(A a, B b) { return float2(a, b); }

template<typename A, typename B>
VEKL_HD constexpr uint2 make_uint2(A a, B b) { return uint2(a, b); }

#define VEKL_VEC_OP_BODY_2_2(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b.y) \
  );

#define VEKL_VEC_OP_BODY_2_3(OP) VEKL_VEC_OP_BODY_2_2(OP)
#define VEKL_VEC_OP_BODY_2_4(OP) VEKL_VEC_OP_BODY_2_2(OP)

#define VEKL_VEC_OP_BODY_3_2(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a.z) \
  );

#define VEKL_VEC_OP_BODY_3_3(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a.z) OP cast_to<RBase>(b.z) \
  );

#define VEKL_VEC_OP_BODY_3_4(OP) VEKL_VEC_OP_BODY_3_3(OP)

#define VEKL_VEC_OP_BODY_4_2(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a.z), \
    cast_to<RBase>(a.w) \
  );

#define VEKL_VEC_OP_BODY_4_3(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a.z) OP cast_to<RBase>(b.z), \
    cast_to<RBase>(a.w) \
  );

#define VEKL_VEC_OP_BODY_4_4(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a.z) OP cast_to<RBase>(b.z), \
    cast_to<RBase>(a.w) OP cast_to<RBase>(b.w) \
  );

#define VEKL_VEC_SCALAR_BODY_2(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b) \
  );

#define VEKL_VEC_SCALAR_BODY_3(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b), \
    cast_to<RBase>(a.z) OP cast_to<RBase>(b) \
  );

#define VEKL_VEC_SCALAR_BODY_4(OP) \
  return RVec( \
    cast_to<RBase>(a.x) OP cast_to<RBase>(b), \
    cast_to<RBase>(a.y) OP cast_to<RBase>(b), \
    cast_to<RBase>(a.z) OP cast_to<RBase>(b), \
    cast_to<RBase>(a.w) OP cast_to<RBase>(b) \
  );

#define VEKL_SCALAR_VEC_BODY_2(OP) \
  return RVec( \
    cast_to<RBase>(a) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a) OP cast_to<RBase>(b.y) \
  );

#define VEKL_SCALAR_VEC_BODY_3(OP) \
  return RVec( \
    cast_to<RBase>(a) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a) OP cast_to<RBase>(b.z) \
  );

#define VEKL_SCALAR_VEC_BODY_4(OP) \
  return RVec( \
    cast_to<RBase>(a) OP cast_to<RBase>(b.x), \
    cast_to<RBase>(a) OP cast_to<RBase>(b.y), \
    cast_to<RBase>(a) OP cast_to<RBase>(b.z), \
    cast_to<RBase>(a) OP cast_to<RBase>(b.w) \
  );

#define VEKL_DEFINE_VEC_VEC_OP(N, M, OP) \
template<typename A, typename B, typename = enable_if_t<(vec_traits<A>::count == N) && (vec_traits<B>::count == M)>> \
VEKL_HD constexpr vec_type_t<typename promote_scalar<typename vec_traits<A>::base, typename vec_traits<B>::base>::type, N> \
operator OP(const A& a, const B& b) { \
  using RBase = typename promote_scalar<typename vec_traits<A>::base, typename vec_traits<B>::base>::type; \
  using RVec = vec_type_t<RBase, N>; \
  VEKL_VEC_OP_BODY_##N##_##M(OP) \
}

#define VEKL_DEFINE_VEC_SCALAR_OP(N, OP) \
template<typename V, typename S, typename = enable_if_t<(vec_traits<V>::count == N) && is_scalar<S>::value>> \
VEKL_HD constexpr vec_type_t<typename promote_scalar<typename vec_traits<V>::base, S>::type, N> \
operator OP(const V& a, S b) { \
  using RBase = typename promote_scalar<typename vec_traits<V>::base, S>::type; \
  using RVec = vec_type_t<RBase, N>; \
  VEKL_VEC_SCALAR_BODY_##N(OP) \
}

#define VEKL_DEFINE_SCALAR_VEC_OP(N, OP) \
template<typename S, typename V, typename = enable_if_t<is_scalar<S>::value && (vec_traits<V>::count == N)>> \
VEKL_HD constexpr vec_type_t<typename promote_scalar<S, typename vec_traits<V>::base>::type, N> \
operator OP(S a, const V& b) { \
  using RBase = typename promote_scalar<S, typename vec_traits<V>::base>::type; \
  using RVec = vec_type_t<RBase, N>; \
  VEKL_SCALAR_VEC_BODY_##N(OP) \
}

#define VEKL_DEFINE_OP_SET(OP) \
  VEKL_DEFINE_VEC_SCALAR_OP(2, OP) \
  VEKL_DEFINE_VEC_SCALAR_OP(3, OP) \
  VEKL_DEFINE_VEC_SCALAR_OP(4, OP) \
  VEKL_DEFINE_SCALAR_VEC_OP(2, OP) \
  VEKL_DEFINE_SCALAR_VEC_OP(3, OP) \
  VEKL_DEFINE_SCALAR_VEC_OP(4, OP) \
  VEKL_DEFINE_VEC_VEC_OP(2, 2, OP) \
  VEKL_DEFINE_VEC_VEC_OP(2, 3, OP) \
  VEKL_DEFINE_VEC_VEC_OP(2, 4, OP) \
  VEKL_DEFINE_VEC_VEC_OP(3, 2, OP) \
  VEKL_DEFINE_VEC_VEC_OP(3, 3, OP) \
  VEKL_DEFINE_VEC_VEC_OP(3, 4, OP) \
  VEKL_DEFINE_VEC_VEC_OP(4, 2, OP) \
  VEKL_DEFINE_VEC_VEC_OP(4, 3, OP) \
  VEKL_DEFINE_VEC_VEC_OP(4, 4, OP)

VEKL_DEFINE_OP_SET(+)
VEKL_DEFINE_OP_SET(-)
VEKL_DEFINE_OP_SET(*)
VEKL_DEFINE_OP_SET(/)

#define VEKL_DEFINE_VEC_VEC_ASSIGN_OP(N, M, OP, OPEQ) \
template<typename A, typename B, typename = enable_if_t<(vec_traits<A>::count == N) && (vec_traits<B>::count == M)>> \
VEKL_HD constexpr A& operator OPEQ(A& a, const B& b) { \
  a = a OP b; \
  return a; \
}

#define VEKL_DEFINE_VEC_SCALAR_ASSIGN_OP(N, OP, OPEQ) \
template<typename V, typename S, typename = enable_if_t<(vec_traits<V>::count == N) && is_scalar<S>::value>> \
VEKL_HD constexpr V& operator OPEQ(V& a, S b) { \
  a = a OP b; \
  return a; \
}

#define VEKL_DEFINE_ASSIGN_OP_SET(OP, OPEQ) \
  VEKL_DEFINE_VEC_SCALAR_ASSIGN_OP(2, OP, OPEQ) \
  VEKL_DEFINE_VEC_SCALAR_ASSIGN_OP(3, OP, OPEQ) \
  VEKL_DEFINE_VEC_SCALAR_ASSIGN_OP(4, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(2, 2, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(2, 3, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(2, 4, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(3, 2, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(3, 3, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(3, 4, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(4, 2, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(4, 3, OP, OPEQ) \
  VEKL_DEFINE_VEC_VEC_ASSIGN_OP(4, 4, OP, OPEQ)

VEKL_DEFINE_ASSIGN_OP_SET(+, +=)
VEKL_DEFINE_ASSIGN_OP_SET(-, -=)
VEKL_DEFINE_ASSIGN_OP_SET(*, *=)
VEKL_DEFINE_ASSIGN_OP_SET(/, /=)

#undef VEKL_DEFINE_ASSIGN_OP_SET
#undef VEKL_DEFINE_VEC_SCALAR_ASSIGN_OP
#undef VEKL_DEFINE_VEC_VEC_ASSIGN_OP

#undef VEKL_DEFINE_OP_SET
#undef VEKL_DEFINE_SCALAR_VEC_OP
#undef VEKL_DEFINE_VEC_SCALAR_OP
#undef VEKL_DEFINE_VEC_VEC_OP
#undef VEKL_SCALAR_VEC_BODY_4
#undef VEKL_SCALAR_VEC_BODY_3
#undef VEKL_SCALAR_VEC_BODY_2
#undef VEKL_VEC_SCALAR_BODY_4
#undef VEKL_VEC_SCALAR_BODY_3
#undef VEKL_VEC_SCALAR_BODY_2
#undef VEKL_VEC_OP_BODY_4_4
#undef VEKL_VEC_OP_BODY_4_3
#undef VEKL_VEC_OP_BODY_4_2
#undef VEKL_VEC_OP_BODY_3_4
#undef VEKL_VEC_OP_BODY_3_3
#undef VEKL_VEC_OP_BODY_3_2
#undef VEKL_VEC_OP_BODY_2_4
#undef VEKL_VEC_OP_BODY_2_3
#undef VEKL_VEC_OP_BODY_2_2

#undef VEKL_DEFINE_VEC4
#undef VEKL_DEFINE_VEC3
#undef VEKL_DEFINE_VEC2
#undef VEKL_ALIGN4
#undef VEKL_ALIGN3
#undef VEKL_ALIGN2
#undef VEKL_VEC4_CTORS
#undef VEKL_VEC3_CTORS
#undef VEKL_VEC2_CTORS

} // namespace vekl

using vekl::uint;

using vekl::float2;
using vekl::float3;
using vekl::float4;

using vekl::int2;
using vekl::int3;
using vekl::int4;

using vekl::uint2;
using vekl::uint3;
using vekl::uint4;

using vekl::bool2;
using vekl::bool3;
using vekl::bool4;

using vekl::make_float2;
using vekl::make_uint2;

using vekl::operator+;
using vekl::operator-;
using vekl::operator*;
using vekl::operator/;
using vekl::operator+=;
using vekl::operator-=;
using vekl::operator*=;
using vekl::operator/=;

#undef VEKL_HD