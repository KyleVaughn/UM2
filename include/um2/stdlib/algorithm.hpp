#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/utility.hpp>

#include <algorithm>

// Contains:
//  clamp
//  copy
//  fill
//  insertionSort
//  max
//  min

namespace um2
{

//==============================================================================
// clamp
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T
{
  return std::clamp(v, lo, hi);
}

#else

template <typename T>
PURE DEVICE constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T
{
  return v < lo ? lo : hi < v ? hi : v;
}

#endif

//==============================================================================
// copy
//==============================================================================
//
// Copies the elements in the range, defined by [first, last), to another range
// beginning at d_first. The function begins by copying *first into *d_first
// and then increments both first and d_first. If first == last, the function
// does nothing.
//
// https://en.cppreference.com/w/cpp/algorithm/copy
//
// We use __restrict__ to tell the compiler that the ranges do not overlap.

template <typename InputIt, typename OutputIt>
HOSTDEV constexpr auto
copy(InputIt __restrict__ first, InputIt last, OutputIt __restrict__ d_first) noexcept
    -> OutputIt
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
// insertionSort
//==============================================================================

template <typename T>
HOSTDEV constexpr void
insertionSort(T * const first, T * const last)
{
  if (first == last) {
    return;
  }
  T * i = first + 1;
  for (; i != last; ++i) {
    T * j = i - 1;
    if (*i < *j) {
      T t = um2::move(*i);
      T * k = j;
      j = i;
      do {
        *j = um2::move(*k);
        j = k;
      } while (j != first && t < *--k);
      *j = um2::move(t);
    }
  }
}

//==============================================================================
// max
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
PURE HOST constexpr auto
max(T x, T y) noexcept -> T
{
  return std::max(x, y);
}

#else

PURE DEVICE constexpr auto
max(float x, float y) noexcept -> float
{
  return ::fmaxf(x, y);
}

PURE DEVICE constexpr auto
max(double x, double y) noexcept -> double
{
  return ::fmax(x, y);
}

template <std::integral T>
PURE DEVICE constexpr auto
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
PURE HOST constexpr auto
min(T x, T y) noexcept -> T
{
  return std::min(x, y);
}

#else

PURE DEVICE constexpr auto
min(float x, float y) noexcept -> float
{
  return ::fminf(x, y);
}

PURE DEVICE constexpr auto
min(double x, double y) noexcept -> double
{
  return ::fmin(x, y);
}

template <std::integral T>
PURE DEVICE constexpr auto
min(T x, T y) noexcept -> T
{
  return ::min(x, y);
}

#endif

} // namespace um2
