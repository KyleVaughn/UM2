#pragma once

#include <um2/math/vec.hpp>

//==============================================================================
// POINT
//==============================================================================
// An alias for a D-dimensional vector. This isn't mathematically correct, but it
// is more efficient to use a vector for a point than to create a separate class
// and deal with the type conversions.

namespace um2
{

//==============================================================================
// Aliases
//==============================================================================

template <Size D, typename T>
using Point = Vec<D, T>;

template <typename T>
using Point1 = Point<1, T>;
template <typename T>
using Point2 = Point<2, T>;
template <typename T>
using Point3 = Point<3, T>;

using Point1f = Point1<float>;
using Point1d = Point1<double>;

using Point2f = Point2<float>;
using Point2d = Point2<double>;

using Point3f = Point3<float>;
using Point3d = Point3<double>;

//==============================================================================
// Constants
//==============================================================================

// Distance between two points, below which they are considered to be equal.
template <std::floating_point T>
inline constexpr T eps_distance = static_cast<T>(1e-6); // 0.1 micron

// Squared distance between two points, below which they are considered to be
// equal.
template <std::floating_point T>
inline constexpr T eps_distance2 = eps_distance<T> * eps_distance<T>;

// Distance between two points, above which they are considered to be infinitely
// far apart. Typically used for invalid points and values.
template <std::floating_point T>
inline constexpr T inf_distance = static_cast<T>(1e10);

//==============================================================================
// Methods
//==============================================================================

template <Size D, class T>
PURE HOSTDEV constexpr auto
midpoint(Point<D, T> a, Point<D, T> const & b) noexcept -> Point<D, T>
{
  // (a + b) / 2
  a += b;
  return a /= 2;
}

template <Size D, class T>
PURE HOSTDEV constexpr auto
isApprox(Point<D, T> const & a, Point<D, T> const & b) noexcept -> bool
{
  return a.squaredDistanceTo(b) < eps_distance2<T>;
}

// Are 3 planar points in counter-clockwise order?
template <class T>
PURE HOSTDEV constexpr auto
areCCW(Point2<T> const & a, Point2<T> const & b, Point2<T> const & c) noexcept -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  T const ab_x = b[0] - a[0];
  T const ab_y = b[1] - a[1];
  T const ac_x = c[0] - a[0];
  T const ac_y = c[1] - a[1];
  // Allow equality, so that we can handle collinear points.
  return 0 <= (ab_x * ac_y - ab_y * ac_x);
}

// Are 3 planar points in counter-clockwise order? We allow for a small amount of
// floating point error.
template <class T>
PURE HOSTDEV constexpr auto
areApproxCCW(Point2<T> const & a, Point2<T> const & b, Point2<T> const & c) noexcept
    -> bool
{
  // 2D cross product, of (b - a) and (c - a).
  T const ab_x = b[0] - a[0];
  T const ab_y = b[1] - a[1];
  T const ac_x = c[0] - a[0];
  T const ac_y = c[1] - a[1];
  return -eps_distance<T> <= (ab_x * ac_y - ab_y * ac_x);
}

} // namespace um2
