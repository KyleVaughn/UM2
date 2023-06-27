#pragma once

#include <um2/geometry/LineSegment.hpp>
#include <um2/math/Mat.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// QUADRILATERAL
// -----------------------------------------------------------------------------
// A 2-dimensional polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 4 vertices are D-dimensional points of type T.

template <typename T>
using Quadrilateral2 = Quadrilateral<2, T>;
using Quadrilateral2f = Quadrilateral2<float>;
using Quadrilateral2d = Quadrilateral2<double>;

template <Size D, typename T>
struct Polytope<2, 1, 4, D, T> {

  Point<D, T> vertices[4];

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
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D, T>;

  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(R /*r*/, S /*s*/) const noexcept
      -> Mat<D, 2, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  edge(Size i) const noexcept -> LineSegment<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;
};

} // namespace um2

#include "Quadrilateral.inl"
