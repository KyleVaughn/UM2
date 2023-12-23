#pragma once

#include <um2/geometry/point.hpp>

namespace um2
{

template <Size D, typename T>
class Ray {

  Point<D, T> _o; // origin
  Vec<D, T> _d;   // direction (unit vector)

  public:

  //============================================================================
  // Constructors
  //============================================================================

  HOSTDEV constexpr Ray(Point<D, T> const & origin, Vec<D, T> const & direction) noexcept
      : _o(origin),
        _d(direction)
  {
    // Check that the direction is a unit vector
    ASSERT(um2::abs(direction.squaredNorm() - static_cast<T>(1)) < static_cast<T>(1e-5));
  }

  //============================================================================
  // Accessors
  //============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  origin() const noexcept -> Point<D, T> const &
  {
    return _o;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  direction() const noexcept -> Vec<D, T> const &
  {
    return _d;
  }

  //============================================================================
  // Methods
  //============================================================================

  HOSTDEV constexpr auto
  operator()(T r) const noexcept -> Point<D, T>
  {
    Point<D, T> res;
    for (Size i = 0; i < D; ++i) {
      res[i] = _o[i] + r * _d[i];
    }
    return res;
  }

}; // class Ray

//==============================================================================
// Aliases
//==============================================================================

template <typename T>
using Ray1 = Ray<1, T>;
template <typename T>
using Ray2 = Ray<2, T>;
template <typename T>
using Ray3 = Ray<3, T>;

using Ray2f = Ray2<float>;
using Ray2d = Ray2<double>;
using Ray3f = Ray3<float>;
using Ray3d = Ray3<double>;

} // namespace um2
