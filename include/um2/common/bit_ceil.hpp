#pragma once

#include <bit>         // std::bit_ceil, std::bit_cast
#include <cassert>     // assert
#include <concepts>    // std::signed_integral, std::unsigned_integral
#include <cstdint>     // std::int*_t
#include <type_traits> // std::make_unsigned_t

// Returns the smallest power of two that is greater than or equal to x. Regardless of
// signedness, the result is always unsigned.

// TODO(kcvaughn@umich.edu): Use [[assume(0 <= x)]] once C++23 is supported.

namespace um2
{

#ifndef __CUDA_ARCH__

template <std::signed_integral T>
constexpr auto bit_ceil(T const x) noexcept -> std::make_unsigned_t<T>
{
  assert(0 <= x && "x must be non-negative");
  return std::bit_ceil(std::bit_cast<std::make_unsigned_t<T>>(x));
}

template <std::unsigned_integral T>
constexpr auto bit_ceil(T const x) noexcept -> T
{
  return std::bit_ceil(x);
}

#else // __CUDA_ARCH__

__device__ constexpr auto bit_ceil(uint32_t const x) noexcept -> uint32_t
{
  if (x <= 1) {
    return x;
  } else {
    return 1 << (32 - __clz(x - 1));
  }
}

__device__ constexpr auto bit_ceil(int32_t const x) noexcept -> uint32_t
{
  assert(0 <= x && "x must be non-negative");
  return bit_ceil(static_cast<uint32_t>(x));
}

__device__ constexpr auto bit_ceil(uint64_t const x) noexcept -> uint64_t
{
  if (x <= 1) {
    return x;
  } else {
    return 1 << (64 - __clzl(x - 1));
  }
}

__device__ constexpr auto bit_ceil(int64_t const x) noexcept -> uint64_t
{
  assert(0 <= x && "x must be non-negative");
  return bit_ceil(static_cast<uint64_t const &>(x));
}

#endif // __CUDA_ARCH__

} // namespace um2
