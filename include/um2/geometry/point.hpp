#pragma once

#include <um2/common/config.hpp>
#include <um2/math/vec.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// POINT
// -----------------------------------------------------------------------------
// An alias for a D-dimensional vector. This isn't technically correct, but it
// is more efficient to use a vector for a point than a separate class.

template <len_t D, typename T>
using Point = Vec<D, T>;

// -- Aliases --

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

// -- Constants --

template <std::floating_point T>
constexpr T point_eps = static_cast<T>(1e-5);

template <std::floating_point T>
constexpr T point_eps_squared = point_eps<T> * point_eps<T>;

template <std::floating_point T>
constexpr T point_inf = static_cast<T>(1e10);

// -- Methods --

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto squaredDistance(Point<D, T> const & a,
                                                    Point<D, T> const & b) -> T
{
  return (a - b).squaredNorm();
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto distance(Point<D, T> const & a, Point<D, T> const & b)
    -> T
{
  return (a - b).norm();
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto midpoint(Point<D, T> const & a, Point<D, T> const & b)
    -> Point<D, T>
{
  return (a + b) / 2;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto isApprox(Point<D, T> const & a, Point<D, T> const & b)
    -> bool
{
  return squaredDistance(a, b) < point_eps_squared<T>;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto areCCW(Point2<T> const & a, Point2<T> const & b,
                                           Point2<T> const & c) -> bool
{
  Point2<T> const ab = b - a;
  Point2<T> const ac = c - a;
  return 0 < cross(ab, ac);
}

} // namespace um2
