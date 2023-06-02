#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/polytope.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// LINE SEGMENT
// -----------------------------------------------------------------------------
// A 1-dimensional polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 2 vertices are D-dimensional points of type T.

template <typename T>
using LineSegment2 = LineSegment<2, T>;
using LineSegment2f = LineSegment2<float>;
using LineSegment2d = LineSegment2<double>;

template <len_t D, typename T>
struct Polytope<1, 1, 2, D, T> {

  Point<D, T> vertices[2];

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) -> Point<D, T> &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const
      -> Point<D, T> const &;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  template <typename R>
  UM2_PURE UM2_HOSTDEV constexpr auto operator()(R r) const noexcept -> Point<D, T>;

  template <typename R>
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto jacobian(R /*r*/) const noexcept
      -> Vec<D, T>;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto length() const noexcept -> T;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool requires(D == 2);
};

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(LineSegment<D, T> const & line) noexcept
    -> AABox<D, T>;

} // namespace um2

#include "line_segment.inl"