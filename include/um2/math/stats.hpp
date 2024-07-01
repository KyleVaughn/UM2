#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/roots.hpp>

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
// sum
//=============================================================================
// For small n, computes the naive sum of the values. For large n, uses
// pairwise summation to minimize floating point error.

namespace sum_detail
{

template <class T>
PURE HOSTDEV constexpr auto
naiveSum(T const * begin, T const * end) noexcept -> T
{
// Very weird false positive here for GCC 12.3 where it thinks "result" is an index
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  T result = *begin;
#pragma GCC diagnostic pop
  while (++begin != end) {
    result += *begin;
  }
  return result;
}

template <class T>
PURE auto
// NOLINTNEXTLINE(misc-no-recursion) OK
pairwiseSum(T const * begin, T const * end) noexcept -> T
{
  auto const n = end - begin;
  ASSERT_ASSUME(n > 0);
  if (n <= 128) {
    return naiveSum(begin, end);
  }
  auto const m = n / 2;
  return pairwiseSum(begin, begin + m) + pairwiseSum(begin + m, end);
}

} // namespace sum_detail

template <class T>
PURE HOSTDEV constexpr auto
sum(T const * begin, T const * end) noexcept -> T
{
#if defined(__CUDA_ARCH__)
  return sum_detail::naiveSum(begin, end);
#else
  return sum_detail::pairwiseSum(begin, end);
#endif
}

//=============================================================================
// mean
//=============================================================================
// Computes the mean of the values in the range [begin, end).

template <class T>
PURE HOSTDEV constexpr auto
mean(T const * begin, T const * end) noexcept -> T
{
  ASSERT_ASSUME(begin != end);
  auto const n = static_cast<T>(end - begin);
  T const vec_sum = um2::sum(begin, end);
  return vec_sum / n;
}

//=============================================================================
// median
//=============================================================================
// Computes the median of the values in the range [begin, end).
// The range must be sorted.

template <class T>
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

// Use Welford's algorithm to compute the variance.
template <class T>
PURE HOSTDEV constexpr auto
variance(T const * begin, T const * end) noexcept -> T
{
  ASSERT_ASSUME(begin != end);
  Int n = 0;
  T mean = 0;
  T m2 = 0;

  while (begin != end) {
    ++n;
    T const delta = *begin - mean;
    mean += delta / static_cast<T>(n);
    m2 += delta * (*begin - mean);
    ++begin;
  }
  ASSERT(n > 1);
  return m2 / static_cast<T>(n - 1);
}

//=============================================================================
// stdDev
//=============================================================================
// Computes the standard deviation of the values in the range [begin, end).

template <class T>
PURE HOSTDEV auto
stdDev(T const * begin, T const * end) noexcept -> T
{
  return um2::sqrt(um2::variance(begin, end));
}

} // namespace um2
