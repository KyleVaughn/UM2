#pragma once

#include <um2/stdlib/math/abs.hpp>
#include <um2/geometry/point.hpp>
#include <um2/common/cast_if_not.hpp>

//==============================================================================
// RAY
//==============================================================================
// A ray is a half-line with an origin and a direction.

namespace um2
{

template <Int D>
class Ray
{

  Point<D> _o;  // origin
  Point<D> _d; // direction (unit vector)

public:
  //============================================================================
  // Constructors
  //============================================================================

  HOSTDEV constexpr Ray(Point<D> const & origin, Point<D> const & direction) noexcept
      : _o(origin),
        _d(direction)
  {
    // Check that the direction is a unit vector
    ASSERT(um2::abs(direction.squaredNorm() - static_cast<Float>(1)) < castIfNot<Float>(1e-5));
  }

  //============================================================================
  // Accessors
  //============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  origin() const noexcept -> Point<D> const &
  {
    return _o;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  direction() const noexcept -> Point<D> const &
  {
    return _d;
  }

  //============================================================================
  // Other member functions
  //============================================================================

  PURE HOSTDEV constexpr auto
  operator()(Float r) const noexcept -> Point<D>
  {
    Point<D> res;
    for (Int i = 0; i < D; ++i) {
      res[i] = _o[i] + r * _d[i];
    }
    return res;
  }

}; // class Ray

//==============================================================================
// Aliases
//==============================================================================

using Ray1 = Ray<1>;
using Ray2 = Ray<2>;
using Ray3 = Ray<3>;

} // namespace um2
