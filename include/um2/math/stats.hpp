#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/roots.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>

//=============================================================================
// STATS
//=============================================================================
// This file contains functions for computing statistics on a range of values.
//
// The following functions are provided:
// mean
// median (requires sorted range)
// variance
// stdDev

namespace um2
{

//=============================================================================
// mean
//=============================================================================
// Computes the mean of the values in the range [begin, end).

template <std::floating_point T>
PURE HOSTDEV constexpr auto
mean(T const * begin, T const * end) noexcept -> T
{
  ASSERT_ASSUME(begin != end);
  auto const n = static_cast<T>(end - begin);
  T result = 0;
  while (begin != end) {
    result += *begin;
    ++begin;
  }
  return result / n;
}

//=============================================================================
// median
//=============================================================================
// Computes the median of the values in the range [begin, end).
// The range must be sorted.

template <std::floating_point T>
PURE HOSTDEV constexpr auto
median(T const * begin, T const * end) noexcept -> T
{
  ASSERT_ASSUME(begin != end);    
  ASSERT(um2::is_sorted(begin, end));    
  auto const size = end - begin;
  auto const * const mid = begin + size / 2;
  // If the size is odd, return the middle element.
  if (size % 2 == 1) {
    return *mid;
  }
  // Otherwise, return the average of the two middle elements.
  return (*mid + *(mid - 1)) / 2;
}

//=============================================================================
// variance
//=============================================================================
// Computes the variance of the values in the range [begin, end).

template <std::floating_point T>
PURE HOSTDEV constexpr auto
variance(T const * begin, T const * end) noexcept -> T
{
  ASSERT_ASSUME(begin != end);
  auto const n_minus_1 = static_cast<T>(end - begin - 1);
  ASSERT(n_minus_1 > 0);
  auto const xbar = um2::mean(begin, end);
  T result = 0;
  while (begin != end) {
    T const x_minus_xbar = *begin - xbar;
    result += x_minus_xbar * x_minus_xbar;
    ++begin;
  }
  return result / n_minus_1;
}

//=============================================================================
// stdDev
//=============================================================================
// Computes the standard deviation of the values in the range [begin, end).

template <std::floating_point T>
PURE HOSTDEV auto
stdDev(T const * begin, T const * end) noexcept -> T
{
  return um2::sqrt(um2::variance(begin, end));
}

} // namespace um2
