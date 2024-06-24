#pragma once

#include <um2/geometry/point.hpp>
#include <um2/math/morton.hpp>
#include <um2/stdlib/numeric/iota.hpp>

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
// as double -> uint32_t, but provides a more accurate sorting of points

namespace um2
{

using MortonCode = uint32_t;

template <class T>
PURE HOSTDEV inline auto
mortonEncode(Point2<T> p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, T>(p[0], p[1]);
}

template <class T>
PURE HOSTDEV inline auto
mortonEncode(Point3<T> const & p) noexcept -> MortonCode
{
  return mortonEncode<MortonCode, T>(p[0], p[1], p[2]);
}

template <class T>
HOSTDEV inline void
mortonDecode(MortonCode morton, Point2<T> & p) noexcept
{
  mortonDecode(morton, p[0], p[1]);
}

template <class T>
HOSTDEV inline void
mortonDecode(MortonCode morton, Point3<T> & p) noexcept
{
  mortonDecode(morton, p[0], p[1], p[2]);
}

template <Int D, class T>
PURE HOSTDEV auto
mortonLess(Point<D, T> const & lhs, Point<D, T> const & rhs) noexcept -> bool
{
  return mortonEncode(lhs) < mortonEncode(rhs);
}

template <Int D, class T>
void
mortonSort(Point<D, T> * const begin, Point<D, T> * const end) noexcept
{
  std::sort(begin, end, mortonLess<D, T>);
}

//==============================================================================
// sortPermutation
//==============================================================================
// Create a permutation that sorts [begin, end) when applied. [begin, end) is
// not modified. scale is used to scale the points to the unit square/cube
// before sorting. If the argument is not provided, the points are assumed to
// be in the unit square/cube.

template <Int D, class T>
void
mortonSortPermutation(Point<D, T> const * begin, Point<D, T> const * end,
                      Int * perm_begin) noexcept

{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
    return mortonEncode(begin[i]) < mortonEncode(begin[j]);
  });
}

template <Int D, class T>
void
mortonSortPermutation(Point<D, T> const * begin, Point<D, T> const * end,
                      Int * perm_begin, Vec<D, T> const scale) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  ASSERT(scale.squaredNorm() > epsDistance2<T>());
  std::sort(perm_begin, perm_begin + n, [&](Int const i, Int const j) {
    return mortonEncode(begin[i] * scale) < mortonEncode(begin[j] * scale);
  });
}
} // namespace um2
