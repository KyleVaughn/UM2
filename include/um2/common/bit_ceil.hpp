#pragma once

#include <um2/common/config.hpp>

#include <bit>         // std::bit_ceil, std::bit_cast
#include <cassert>     // assert
#include <concepts>    // std::signed_integral, std::unsigned_integral
#include <cstdint>     // std::int*_t
#include <type_traits> // std::make_unsigned_t

// -----------------------------------------------------------------------------
// Returns the smallest power of two that is greater than or equal to x.
// -----------------------------------------------------------------------------

// TODO(kcvaughn@umich.edu): Use [[assume(0 <= x)]] once C++23 is supported.

// We disable warnings about lower_case function names because we want to match the
// names of the functions in the standard library.
namespace um2
{

#ifndef __CUDA_ARCH__

template <std::unsigned_integral T>
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr auto bit_ceil(T const x) noexcept -> T
{
  return std::bit_ceil(x);
}

#else // __CUDA_ARCH__

// NOLINTNEXTLINE(readability-identifier-naming)
__device__ constexpr auto bit_ceil(uint32_t const x) noexcept -> uint32_t
{
  if (x <= 1) {
    return 1;
  } else {
    return 1 << (32 - __clz(x - 1));
  }
}

// NOLINTNEXTLINE(readability-identifier-naming)
__device__ constexpr auto bit_ceil(uint64_t const x) noexcept -> uint64_t
{
  if (x <= 1) {
    return 1;
  } else {
    return 1 << (64 - __clzll(x - 1));
  }
}

#endif // __CUDA_ARCH__

template <std::signed_integral T>
// NOLINTNEXTLINE(readability-identifier-naming)
UM2_HOSTDEV constexpr auto bit_ceil(T const x) noexcept -> T
{
  assert(0 <= x && "x must be non-negative");
  return static_cast<T>(bit_ceil(static_cast<std::make_unsigned_t<T>>(x)));
}

} // namespace um2
