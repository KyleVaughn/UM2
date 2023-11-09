#pragma once

#include <um2/geometry/point.hpp>

namespace um2
{

template <Size D, typename T>
struct Ray {

  Point<D, T> o; // origin
  Vec<D, T> d;   // direction (unit vector)

  //============================================================================
  // Constructors
  //============================================================================

  HOSTDEV constexpr Ray(Point<D, T> const & origin, Vec<D, T> const & direction) noexcept
      : o(origin),
      d(direction)
  {
    ASSERT(um2::abs(direction.squaredNorm() - static_cast<T>(1)) < static_cast<T>(1e-5));
  }

  //============================================================================
  // Methods
  //============================================================================

  HOSTDEV constexpr auto
  operator()(T r) const noexcept -> Point<D, T>
  {
    Point<D, T> res;
    for (Size i = 0; i < D; ++i) {
      res[i] = o[i] + r * d[i];
    }
    return res;
  }

}; // struct Ray

//==============================================================================
// Aliases
//==============================================================================

template <typename T>
using Ray1 = Ray<1, T>;
template <typename T>
using Ray2 = Ray<2, T>;

using Ray2f = Ray2<float>;
using Ray2d = Ray2<double>;

} // namespace um2
