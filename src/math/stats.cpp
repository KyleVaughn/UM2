#include <um2/math/stats.hpp>

#include <um2/stdlib/assert.hpp>

namespace um2
{

//=============================================================================
// mean
//=============================================================================

PURE HOSTDEV auto
mean(Float const * begin, Float const * end) noexcept -> Float
{
  ASSERT_ASSUME(begin != end);
  auto const n = static_cast<Float>(end - begin);
  Float result = 0;
  while (begin != end) {
    result += *begin;
    ++begin;
  }
  return result / n;
}

//=============================================================================
// median
//=============================================================================

PURE HOSTDEV auto
median(Float const * begin, Float const * end) noexcept -> Float
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

PURE HOSTDEV auto
variance(Float const * begin, Float const * end) noexcept -> Float
{
  ASSERT_ASSUME(begin != end);
  auto const n_minus_1 = static_cast<Float>(end - begin - 1);
  ASSERT(n_minus_1 > 0);
  auto const xbar = um2::mean(begin, end);
  Float result = 0;
  while (begin != end) {
    Float const x_minus_xbar = *begin - xbar;
    result += x_minus_xbar * x_minus_xbar;
    ++begin;
  }
  return result / n_minus_1;
}

//=============================================================================
// stdDev
//=============================================================================

PURE HOSTDEV auto
stdDev(Float const * begin, Float const * end) noexcept -> Float
{
  return um2::sqrt(um2::variance(begin, end));
}

} // namespace um2
