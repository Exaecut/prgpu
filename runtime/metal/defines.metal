#pragma once

namespace vekl {
    template <typename T> using device_ptr = device T*;
    template <typename T> using device_cptr = device const T*;
    template <typename T> using constant_ref = constant T&;
}

#define device_ptr(T) ::vekl::device_ptr<T>
#define device_cptr(T) ::vekl::device_cptr<T>
#define constant_ref(T) ::vekl::constant_ref<T>

#define restrict_ptr restrict
#define threadgroup_mem threadgroup
#define thread_local thread

#define param_ro(T, name, slot) device_cptr(T) name [[buffer(slot)]]
#define param_rw(T, name, slot) device_ptr(T) name [[buffer(slot)]]
#define param_wo(T, name, slot) device_ptr(T) name [[buffer(slot)]]
#define param_cbuf(T, name, slot) constant_ref(T) name [[buffer(slot)]]

#define thread_pos_param(name) uint2 name [[thread_position_in_grid]]
#define thread_pos_init(name)

#define threadgroup_barrier_all() threadgroup_barrier(mem_flags::mem_threadgroup)