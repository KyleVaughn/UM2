#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/math.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <numeric>

namespace um2
{

//=============================================================================
// mean
//=============================================================================

template <std::floating_point T>
constexpr auto
mean(T const * begin, T const * end) -> T
{
  assert(begin != end);
  return std::reduce(begin, end, static_cast<T>(0)) / static_cast<T>(end - begin);
}

//=============================================================================
// median
//=============================================================================

template <std::floating_point T>
constexpr auto
median(T const * begin, T const * end) -> T
{
  assert(begin != end);
  assert(std::is_sorted(begin, end));
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
constexpr auto
variance(T const * begin, T const * end) -> T
{
  assert(begin != end);
  auto const m = mean(begin, end);
  return std::accumulate(
             begin, end, static_cast<T>(0),
             [m](auto const acc, auto const x) { return acc + (x - m) * (x - m); }) /
         static_cast<T>(end - begin - 1);
}

//=============================================================================
// stdDev
//=============================================================================

template <std::floating_point T>
constexpr auto
stdDev(T const * begin, T const * end) -> T
{
  return um2::sqrt(variance(begin, end));
}

} // namespace um2
