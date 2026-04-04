#pragma once

#include "config.h"

// Minimal compile-time traits to support vector constructors and math dispatch logic.
namespace vekl {
    namespace traits {
        template<bool B, class T = void>
        struct enable_if {};

        template<class T>
        struct enable_if<true, T> { using type = T; };

        template<typename T, typename U>
        struct is_same { static constexpr bool value = false; };

        template<typename T>
        struct is_same<T, T> { static constexpr bool value = true; };

        template <typename T>
        struct remove_reference { using type = T; };

        template <typename T>
        struct remove_reference<T&> { using type = T; };

        template <typename T>
        struct remove_reference<T&&> { using type = T; };

        template <typename T>
        struct identity {
            using type = T;
        };

        template<typename T>
        struct is_float_type {
            static constexpr bool value = false;
        };

        template<>
        struct is_float_type<float> {
            static constexpr bool value = true;
        };

        template<typename T>
        struct is_int_type {
            static constexpr bool value = false;
        };

        template<>
        struct is_int_type<int> {
            static constexpr bool value = true;
        };

        template<typename T>
        struct is_uint_type {
            static constexpr bool value = false;
        };

        template<>
        struct is_uint_type<unsigned int> {
            static constexpr bool value = true;
        };

        template<typename T>
        struct is_bool_type {
            static constexpr bool value = false;
        };

        template<>
        struct is_bool_type<bool> {
            static constexpr bool value = true;
        };
    }
}