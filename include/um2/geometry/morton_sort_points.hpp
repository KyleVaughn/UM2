#pragma once

#include <um2/geometry/point.hpp>
#include <um2/math/morton.hpp>

#include <um2/stdlib/algorithm.hpp>

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

PURE HOSTDEV auto
mortonEncode(Point2 const & p) noexcept -> MortonCode; 

PURE HOSTDEV auto
mortonEncode(Point3 const & p) noexcept -> MortonCode; 

HOSTDEV void
mortonDecode(MortonCode morton, Point2 & p) noexcept;

HOSTDEV void
mortonDecode(MortonCode morton, Point3 & p) noexcept;

template <Size D>
PURE HOSTDEV auto
mortonLess(Point<D> const & lhs, Point<D> const & rhs) noexcept -> bool
{
  return mortonEncode(lhs) < mortonEncode(rhs);
}

template <Size D>
void
mortonSort(Point<D> * const begin, Point<D> * const end) noexcept
{
  std::sort(begin, end, mortonLess<D>);
}

} // namespace um2
