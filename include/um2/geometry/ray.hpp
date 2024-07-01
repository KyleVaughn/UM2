#pragma once

#include <um2/common/cast_if_not.hpp>
#include <um2/geometry/point.hpp>
#include <um2/stdlib/math/abs.hpp>

//==============================================================================
// RAY
//==============================================================================
// A ray is a half-line with an origin and a direction.

namespace um2
{

template <Int D, class T>
class Ray
{

  Point<D, T> _o; // origin
  Point<D, T> _d; // direction (unit vector)

public:
  //============================================================================
  // Constructors
  //============================================================================

  HOSTDEV constexpr Ray(Point<D, T> const & origin,
                        Point<D, T> const & direction) noexcept
      : _o(origin),
        _d(direction)
  {
    // Check that the direction is a unit vector
    ASSERT(um2::abs(direction.squaredNorm() - static_cast<T>(1)) < castIfNot<T>(1e-5));
  }

  //============================================================================
  // Accessors
  //============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  origin() const noexcept -> Point<D, T> const &
  {
    return _o;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  direction() const noexcept -> Point<D, T> const &
  {
    return _d;
  }

  //============================================================================
  // Other member functions
  //============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  operator()(T r) const noexcept -> Point<D, T>
  {
    return _o + r * _d;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  inverseDirection() const noexcept -> Point<D, T>
  {
    return 1 / _d;
  }

}; // class Ray

//==============================================================================
// Aliases
//==============================================================================

template <class T>
using Ray1 = Ray<1, T>;

template <class T>
using Ray2 = Ray<2, T>;

template <class T>
using Ray3 = Ray<3, T>;

using Ray2F = Ray2<Float>;
using Ray3F = Ray3<Float>;

} // namespace um2
