#pragma once

#include <um2/geometry/point.hpp>
#include <um2/math/morton.hpp>

#include <algorithm> // std::sort

//==============================================================================
// Morton encoding/decoding
//==============================================================================
// Morton encoding and decoding for points in the unit square/cube.
//
// Valid mappings without loss of precision are:
// float -> uint32_t
// double -> uint64_t or uint32_t
//
// On CPU, the double -> uint64_t mapping is used is approximately as performant
// as double -> uint32_t, but provides a more accurate sorting of points, hence
// we only provide the double -> uint64_t.

namespace um2
{

template <Size D>
PURE HOSTDEV auto
mortonEncode(Point<D> const & p) noexcept -> U
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
