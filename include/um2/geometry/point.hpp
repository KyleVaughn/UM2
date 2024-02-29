#pragma once

#include <um2/common/cast_if_not.hpp>
#include <um2/math/simd.hpp>
#include <um2/math/vec.hpp>

//==============================================================================
// POINT
//==============================================================================
// An alias for a D-dimensional vector of type F. This isn't mathematically
// correct, but it is more efficient to use a vector for a point than to create
// a separate class and deal with the type conversions.

namespace um2
{

//==============================================================================
// Aliases
//==============================================================================

template <Int D>
using Point = Vec<D, Float>;

using Point1 = Point<1>;

#if UM2_ENABLE_SIMD_P2
using Point2 = SIMD<2, Float>; 
#else
using Point2 = Vec<2, Float>;
#endif
using Point3 = Point<3>;

//==============================================================================
// Constants
//==============================================================================
// eps_distance:
//   Distance between two points, below which they are considered to be equal.
// eps_distance2:
//   Squared distance between two points, below which they are considered to be
//   equal.
// inf_distance:
//  Distance between two points, above which they are considered to be
//  infinitely far apart. Typically used for invalid points and values.
//
// NOTE: fast-math assumes no infinities, so we need inf_distance to be finite.

inline constexpr Float eps_distance = castIfNot<Float>(1e-6); // 0.1 micron
inline constexpr Float eps_distance2 = castIfNot<Float>(1e-12);
inline constexpr Float inf_distance = castIfNot<Float>(1e8); // 1000 km

//==============================================================================
// Functions 
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
squaredDistance(Point<D> const & a, Point<D> const & b) noexcept -> Float
{
  return squaredNorm(a - b);
}

template <Int D>
PURE HOSTDEV constexpr auto
distance(Point<D> const & a, Point<D> const & b) noexcept -> Float
{
  return um2::sqrt(squaredDistance(a, b)); 
}

template <Int D>
PURE HOSTDEV constexpr auto
midpoint(Point<D> a, Point<D> const & b) noexcept -> Point<D>
{
  // (a + b) / 2
  a += b;
  return a /= 2;
}

template <Int D>
PURE HOSTDEV constexpr auto
isApprox(Point<D> const & a, Point<D> const & b) noexcept -> bool
{
  return squaredDistance(a, b) < eps_distance2;
}

// Are 3 planar points in counter-clockwise order?
PURE HOSTDEV constexpr auto
areCCW(Point2 const & a, Point2 const & b, Point2 const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab = b - a;
  auto const ac = c - a;
  // Allow equality, so that we can handle collinear points.
  return 0 <= cross(ab, ac);
}

// Are 3 planar points in counter-clockwise order? We allow for a small amount of
// floating point error.
PURE HOSTDEV constexpr auto
areApproxCCW(Point2 const & a, Point2 const & b, Point2 const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab = b - a;
  auto const ac = c - a;
  return -eps_distance <= cross(ab, ac); 
}

template <typename T>
PURE HOSTDEV constexpr auto
makePoint(T x, T y) noexcept -> Point2
{
  Point2 p;
  p[0] = x;
  p[1] = y;
  return p; 
}

} // namespace um2
