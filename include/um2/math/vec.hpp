#pragma once

#include <type_traits> // std::is_arithmetic_v

namespace um2
{

// -----------------------------------------------------------------------------
// VEC
// -----------------------------------------------------------------------------
// A D-dimensional vector with arithmetic data type T.

template <len_t D, typename T>
requires(std::is_arithmetic_v<T>) struct Vec {
}; // struct Vec

} // namespace um2