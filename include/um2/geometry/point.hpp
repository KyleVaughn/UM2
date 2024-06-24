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

template <Int D, class T>
using Point = Vec<D, T>;

template <class T>
using Point1 = Point<1, T>;

template <class T>
using Point2 = Point<2, T>;

template <class T>
using Point3 = Point<3, T>;

using Point2f = Point2<float>;
using Point2d = Point2<double>;

//==============================================================================
// Functions
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
midpoint(Point<D, T> a, Point<D, T> const & b) noexcept -> Point<D, T>
{
  // (a + b) / 2
  a += b;
  return a /= 2;
}

// Check if 3 planar points in counter-clockwise order
template <class T>
PURE HOSTDEV constexpr auto
areCCW(Point2<T> const & a, Point2<T> const & b, Point2<T> const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab = b - a;
  auto const ac = c - a;
  // Allow equality, so that we can handle collinear points.
  return 0 <= ab.cross(ac);
}

// Check if 3 planar points in counter-clockwise order. Allow for a small amount of
// floating point error.
template <class T>
PURE HOSTDEV constexpr auto
areApproxCCW(Point2<T> const & a, Point2<T> const & b,
             Point2<T> const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  auto const ab = b - a;
  auto const ac = c - a;
  return -epsDistance<T>() <= ab.cross(ac);
}

} // namespace um2
