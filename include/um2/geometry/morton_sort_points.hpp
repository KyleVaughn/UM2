#pragma once

#include <um2/geometry/point.hpp>
#include <um2/math/morton.hpp>

#include <algorithm> // std::sort

//==============================================================================
// Morton encoding/decoding
//==============================================================================
// Morton encoding and decoding for points in the unit square/cube.

namespace um2
{

template <std::unsigned_integral U, Size D, std::floating_point T>
PURE HOSTDEV auto
mortonEncode(Point<D, T> const & p) noexcept -> U
{
  if constexpr (D == 2) {
    return mortonEncode<U>(p[0], p[1]);
  } else if constexpr (D == 3) {
    return mortonEncode<U>(p[0], p[1], p[2]);
  } else {
    static_assert(D == 2 || D == 3);
    return 0;
  }
}

template <std::unsigned_integral U, Size D, std::floating_point T>
HOSTDEV void
mortonDecode(U const morton, Point<D, T> & p) noexcept
{
  if constexpr (D == 2) {
    mortonDecode(morton, p[0], p[1]);
  } else if constexpr (D == 3) {
    mortonDecode(morton, p[0], p[1], p[2]);
  } else {
    static_assert(D == 2 || D == 3);
  }
}

template <std::unsigned_integral U, Size D, std::floating_point T>
PURE HOSTDEV auto
mortonLess(Point<D, T> const & lhs, Point<D, T> const & rhs) noexcept -> bool
{
  return mortonEncode<U>(lhs) < mortonEncode<U>(rhs);
}

template <std::unsigned_integral U, Size D, std::floating_point T>
void
mortonSort(Point<D, T> * const begin, Point<D, T> * const end) noexcept
{
  std::sort(begin, end, mortonLess<U, D, T>);
}

} // namespace um2
