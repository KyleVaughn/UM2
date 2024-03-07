#pragma once

#include <um2/geometry/point.hpp>
#include <um2/math/morton.hpp>
#include <um2/stdlib/numeric.hpp>

#include <algorithm>

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

#if UM2_ENABLE_FLOAT64
using MortonCode = uint64_t;
#else
using MortonCode = uint32_t;
#endif

PURE HOSTDEV inline auto
mortonEncode(Point2 p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, Float>(p[0], p[1]);
}

PURE HOSTDEV inline auto
mortonEncode(Point3 const & p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, Float>(p[0], p[1], p[2]);
}

HOSTDEV inline void
mortonDecode(MortonCode morton, Point2 & p) noexcept
{
  mortonDecode(morton, p[0], p[1]);
}

HOSTDEV inline void
mortonDecode(MortonCode morton, Point3 & p) noexcept
{
  mortonDecode(morton, p[0], p[1], p[2]);
}

template <Int D>
PURE HOSTDEV auto
mortonLess(Point<D> const & lhs, Point<D> const & rhs) noexcept -> bool
{
  return mortonEncode(lhs) < mortonEncode(rhs);
}

template <Int D>
void
mortonSort(Point<D> * const begin, Point<D> * const end) noexcept
{
  std::sort(begin, end, mortonLess<D>);
}

//==============================================================================
// sortPermutation
//==============================================================================
// Create a permutation that sorts [begin, end) when applied. [begin, end) is
// not modified. scale is used to scale the points to the unit square/cube
// before sorting. If the argument is not provided, the points are assumed to
// be in the unit square/cube.

template <Int D>
void
mortonSortPermutation(Point<D> const * begin,
                      Point<D> const * end,
                      Int * perm_begin,
                      Vec<D, Float> const & scale = Vec<D, Float>::zero()) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  bool const has_scale = scale.squaredNorm() > eps_distance2;
  if (has_scale) {
    std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
      return mortonEncode(begin[i] * scale) < mortonEncode(begin[j] * scale);
    });
  } else {
    std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
      return mortonEncode(begin[i]) < mortonEncode(begin[j]);
    });
  }
}
} // namespace um2
