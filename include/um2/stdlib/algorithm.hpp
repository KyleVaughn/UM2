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

template <typename InputIt, typename OutputIt>
HOSTDEV constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  for (; first != last; ++first, ++d_first) {
    *d_first = *first;
  }
  return d_first;
}

//==============================================================================
// fill
//==============================================================================

template <typename ForwardIt, typename T>
HOSTDEV constexpr void
fill(ForwardIt first, ForwardIt last, T const & value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

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
