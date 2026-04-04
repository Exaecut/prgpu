#pragma once

namespace vekl {
    template <typename T> using device_ptr = T*;
    template <typename T> using device_cptr = const T*;
    template <typename T> using constant_ref = const T&;
}

#define device_ptr(T) ::vekl::device_ptr<T>
#define device_cptr(T) ::vekl::device_cptr<T>
#define constant_ref(T) ::vekl::constant_ref<T>

#define kernel
#define device
#define restrict_ptr __restrict__
#define constant const
#define threadgroup_mem

#define param_ro(T, name, slot) const device_cptr(T) name
#define param_rw(T, name, slot) device_ptr(T) name
#define param_wo(T, name, slot) device_ptr(T) name
#define param_cbuf(T, name, slot) constant_ref(T) name

#define thread_pos_param(name) ::vekl::uint2 name
#define thread_pos_init(name)

#define threadgroup_barrier_all()