#pragma once

#include <um2/geometry/LineSegment.hpp>
#include <um2/math/Mat.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// TRIANGLE
// -----------------------------------------------------------------------------
// A 2-dimensional polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 3 vertices are D-dimensional points of type T.

template <typename T>
using Triangle2 = Triangle<2, T>;
using Triangle2f = Triangle2<float>;
using Triangle2d = Triangle2<double>;

template <Size D, typename T>
struct Polytope<2, 1, 3, D, T> {

  Point<D, T> vertices[3];

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Point<D, T> &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Point<D, T> const &;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  operator()(R r, S s) const noexcept -> Point<D, T>;

  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(R /*r*/, S /*s*/) const noexcept
      -> Mat<D, 2, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  edge(Size i) const noexcept -> LineSegment<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool requires(D == 2);
};

} // namespace um2

#include "Triangle.inl"
