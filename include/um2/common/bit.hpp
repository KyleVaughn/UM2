#pragma once

#include <um2/common/config.hpp>

#include <bit>         // std::bit_cast, std::bit_width, std::bit_ceil
#include <cassert>     // assert
#include <concepts>    // std::signed_integral, std::unsigned_integral
#include <cstdint>     // std::int*_t
#include <type_traits> // std::make_unsigned_t, std::is_trivially_constructible_v, etc.

// -----------------------------------------------------------------------------
// A GPU-compatible implementation of some of the functions in <bit>.
// -----------------------------------------------------------------------------

// We disable warnings about lower_case function names because we want to match the
// names of the functions in the standard library.
// NOLINTBEGIN(readability-identifier-naming)
namespace um2
{

// -----------------------------------------------------------------------------
// bit_cast
// -----------------------------------------------------------------------------
// Obtain a value of type To by reinterpreting the object representation of From.

#ifndef __CUDA_ARCH__
template <typename To, typename From>
constexpr auto bit_cast(From const & from) noexcept -> To
{
  return std::bit_cast<To>(from);
}
#else  // __CUDA_ARCH__
// (kcvaughn): I'm not sure this is legal C++ from a constexpr perspective, but it
// seems to work for CUDA.
template <typename To, typename From>
requires(sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
         std::is_trivially_copyable_v<To>) __device__
    constexpr auto bit_cast(From const & from) noexcept -> To
{
  static_assert(std::is_trivially_constructible_v<To>);
  To to;
  memcpy(&to, &from, sizeof(To));
  return to;
}
#endif // __CUDA_ARCH__

// -----------------------------------------------------------------------------
// bit_width
// -----------------------------------------------------------------------------
// Returns the number of bits required to represent the value of x. Returns 0 if
// x is 0.

#ifndef __CUDA_ARCH__
template <std::unsigned_integral T>
UM2_CONST constexpr auto bit_width(T const x) noexcept -> T
{
  return std::bit_width(x);
}
#else  // __CUDA_ARCH__
UM2_CONST __device__ constexpr auto bit_width(uint32_t const x) noexcept -> uint32_t
{
  return (x == 0) ? 0 : 32 - __clz(x);
}

UM2_CONST __device__ constexpr auto bit_width(uint64_t const x) noexcept -> uint64_t
{
  return (x == 0) ? 0 : 64 - __clzll(x);
}
#endif // __CUDA_ARCH__

// Define for signed types.
template <std::signed_integral T>
UM2_NDEBUG_CONST UM2_HOSTDEV constexpr auto bit_width(T const x) noexcept -> T
{
  assert(0 <= x && "x must be non-negative");
  return static_cast<T>(bit_width(static_cast<std::make_unsigned_t<T>>(x)));
}

// -----------------------------------------------------------------------------
// bit_ceil
// -----------------------------------------------------------------------------

#ifndef __CUDA_ARCH__
template <std::unsigned_integral T>
UM2_CONST constexpr auto bit_ceil(T const x) noexcept -> T
{
  return std::bit_ceil(x);
}
#else  // __CUDA_ARCH__
template <std::unsigned_integral T>
UM2_CONST __device__ constexpr auto bit_ceil(T const x) noexcept -> T
{
  return (x <= 1) ? 1 : 1 << bit_width(x - 1);
}
#endif // __CUDA_ARCH__

// Define for signed types.
template <std::signed_integral T>
UM2_HOSTDEV constexpr auto bit_ceil(T const x) noexcept -> T
{
  assert(0 <= x && "x must be non-negative");
  return static_cast<T>(bit_ceil(static_cast<std::make_unsigned_t<T>>(x)));
}

} // namespace um2
// NOLINTEND(readability-identifier-naming)
