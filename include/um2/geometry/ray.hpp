#pragma once

#include <um2/geometry/point.hpp>

//==============================================================================
// RAY
//==============================================================================
// A ray is a half-line with an origin and a direction.

namespace um2
{

template <I D>
class Ray
{

  Point<D> _o;  // origin
  Vec<D, F> _d; // direction (unit vector)

public:
  //============================================================================
  // Constructors
  //============================================================================

  HOSTDEV constexpr Ray(Point<D> const & origin, Vec<D, F> const & direction) noexcept
      : _o(origin),
        _d(direction)
  {
    // Check that the direction is a unit vector
    ASSERT(um2::abs(direction.squaredNorm() - static_cast<F>(1)) < condCast<F>(1e-5));
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
  direction() const noexcept -> Vec<D, F> const &
  {
    return _d;
  }

  //============================================================================
  // Methods
  //============================================================================

  PURE HOSTDEV constexpr auto
  operator()(F r) const noexcept -> Point<D>
  {
    Point<D> res;
    for (I i = 0; i < D; ++i) {
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
