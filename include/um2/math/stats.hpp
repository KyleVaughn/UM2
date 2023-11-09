#pragma once

#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/math.hpp>

namespace um2
{

//=============================================================================
// mean
//=============================================================================

template <std::floating_point T>
HOSTDEV constexpr auto
mean(T const * begin, T const * end) -> T
{
  ASSERT_ASSUME(begin != end);
  T const n = static_cast<T>(end - begin);
  T result = static_cast<T>(0);
  while (begin != end) {
    result += *begin;
    ++begin;
  }
  return result / n;
}

//=============================================================================
// median
//=============================================================================

template <std::floating_point T>
HOSTDEV constexpr auto
median(T const * begin, T const * end) -> T
{
  ASSERT_ASSUME(begin != end);
  ASSERT(um2::is_sorted(begin, end));
  auto const size = end - begin;
  auto const mid = begin + size / 2;
  // If the size is odd, return the middle element.
  if (size % 2 == 1) {
    return *mid;
  }
  // Otherwise, return the average of the two middle elements.
  return (*mid + *(mid - 1)) / static_cast<T>(2);
}

//=============================================================================
// variance
//=============================================================================

template <std::floating_point T>
HOSTDEV constexpr auto
variance(T const * begin, T const * end) -> T
{
  ASSERT_ASSUME(begin != end);
  T const n_minus_1 = static_cast<T>(end - begin - 1);
  auto const xbar = um2::mean(begin, end);
  T result = static_cast<T>(0);
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

template <std::floating_point T>
HOSTDEV constexpr auto
stdDev(T const * begin, T const * end) -> T
{
  return um2::sqrt(um2::variance(begin, end));
}

} // namespace um2
