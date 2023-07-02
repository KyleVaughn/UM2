#pragma once

#include <um2/geometry/Point.hpp>

namespace um2
{

template <Size D, typename T>
struct Ray {

  Point<D, T> o; // origin
  Vec<D, T> d; // direction (unit vector)

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  HOSTDEV constexpr Ray(Point<D, T> const & origin, Vec<D, T> const & direction) noexcept;

}; // struct Ray

// -- Aliases --

template <typename T>
using Ray1 = Ray<1, T>;

template <typename T>
using Ray2 = Ray<2, T>;

using Ray2f = Ray2<float>;
using Ray2d = Ray2<double>;

} // namespace um2

#include "Ray.inl"
