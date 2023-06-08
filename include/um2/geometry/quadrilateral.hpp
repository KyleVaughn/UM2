#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/math/mat.hpp>

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

template <len_t D, typename T>
struct Polytope<2, 1, 4, D, T> {

  Point<D, T> vertices[4];

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) -> Point<D, T> &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const
      -> Point<D, T> const &;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  template <typename R, typename S>
  UM2_PURE UM2_HOSTDEV constexpr auto operator()(R r, S s) const noexcept -> Point<D, T>;

  template <typename R, typename S>
  UM2_PURE UM2_HOSTDEV
      [[nodiscard]] constexpr auto jacobian(R /*r*/, S /*s*/) const noexcept
      -> Mat<D, 2, T>;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto edge(len_t i) const noexcept
      -> LineSegment<D, T>;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool
    requires(D == 2);
};

} // namespace um2

#include "quadrilateral.inl"
