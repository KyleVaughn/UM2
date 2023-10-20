#pragma once

#include <um2/stdlib/utility.hpp>

#include <algorithm>

// Contains:
//  clamp
//  copy
//  fill
//  max
//  min

namespace um2
{

//==============================================================================
// clamp
//==============================================================================
//
// We deviate from the std::clamp implementation, passing the arguments by
// value instead of by const reference for fundamental types.

template <um2::fundamental T>
CONST HOSTDEV constexpr auto
clamp(T v, T lo, T hi) noexcept -> T
{
  return v < lo ? lo : (hi < v ? hi : v);
}

//==============================================================================
// copy
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename InputIt, typename OutputIt>
HOST constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  // std::copy optimizes to memmove when possible.
  // False positive of memory leak here.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) justified above
  return std::copy(first, last, d_first);
}

#else

template <typename InputIt, typename OutputIt>
DEVICE constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  while (first != last) {
    *d_first = *first;
    ++first;
    ++d_first;
  }
  return d_first;
}

#endif

//==============================================================================
// fill
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename ForwardIt, typename T>
HOST constexpr void
fill(ForwardIt first, ForwardIt last, T const & value)
{
  std::fill(first, last, value);
}

#else

template <typename ForwardIt, typename T>
DEVICE constexpr void
fill(ForwardIt first, ForwardIt last, T const & value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

#endif

//==============================================================================
// max
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
CONST HOST constexpr auto
max(T x, T y) noexcept -> T
{
  return std::max(x, y);
}

#else

CONST DEVICE constexpr auto
max(float x, float y) noexcept -> float
{
  return ::fmaxf(x, y);
}

CONST DEVICE constexpr auto
max(double x, double y) noexcept -> double
{
  return ::fmax(x, y);
}

template <std::integral T>
CONST DEVICE constexpr auto
max(T x, T y) noexcept -> T
{
  return ::max(x, y);
}

#endif

//==============================================================================
// min
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
CONST HOST constexpr auto
min(T x, T y) noexcept -> T
{
  return std::min(x, y);
}

#else

CONST DEVICE constexpr auto
min(float x, float y) noexcept -> float
{
  return ::fminf(x, y);
}

CONST DEVICE constexpr auto
min(double x, double y) noexcept -> double
{
  return ::fmin(x, y);
}

template <std::integral T>
CONST DEVICE constexpr auto
min(T x, T y) noexcept -> T
{
  return ::min(x, y);
}

#endif

} // namespace um2
