#pragma once

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

template <I D>
using Point = Vec<D, F>;

using Point1 = Point<1>;
using Point2 = Point<2>;
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

inline constexpr F eps_distance = condCast<F>(1e-6); // 0.1 micron
inline constexpr F eps_distance2 = condCast<F>(1e-12);
inline constexpr F inf_distance = condCast<F>(1e8); // 1000 km

//==============================================================================
// Methods
//==============================================================================

template <I D>
PURE HOSTDEV constexpr auto
midpoint(Point<D> a, Point<D> const & b) noexcept -> Point<D>
{
  // (a + b) / 2
  a += b;
  return a /= 2;
}

template <I D>
PURE HOSTDEV constexpr auto
isApprox(Point<D> const & a, Point<D> const & b) noexcept -> bool
{
  return a.squaredDistanceTo(b) < eps_distance2;
}

// Are 3 planar points in counter-clockwise order?
PURE HOSTDEV constexpr auto
areCCW(Point2 const & a, Point2 const & b, Point2 const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab_x = b[0] - a[0];
  auto const ab_y = b[1] - a[1];
  auto const ac_x = c[0] - a[0];
  auto const ac_y = c[1] - a[1];
  // Allow equality, so that we can handle collinear points.
  return 0 <= (ab_x * ac_y - ab_y * ac_x);
}

// Are 3 planar points in counter-clockwise order? We allow for a small amount of
// floating point error.
PURE HOSTDEV constexpr auto
areApproxCCW(Point2 const & a, Point2 const & b, Point2 const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab_x = b[0] - a[0];
  auto const ab_y = b[1] - a[1];
  auto const ac_x = c[0] - a[0];
  auto const ac_y = c[1] - a[1];
  return -eps_distance <= (ab_x * ac_y - ab_y * ac_x);
}

} // namespace um2
