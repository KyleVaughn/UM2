#pragma once

#include <um2/geometry/Polygon.hpp>
#include <um2/geometry/Triangle.hpp>

namespace um2
{

//==============================================================================
// QUADRILATERAL
//==============================================================================
//
// A 2-polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 4 vertices are D-dimensional points of type T.

template <Size D, typename T>
struct Polytope<2, 1, 4, D, T> {

  Point<D, T> v[4];

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  HOSTDEV constexpr Polytope(Point<D, T> const & p0, Point<D, T> const & p1,
                             Point<D, T> const & p2, Point<D, T> const & p3) noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Point<D, T> &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Point<D, T> const &;

  //==============================================================================
  // Methods
  //==============================================================================

  template <typename R, typename S>
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D, T>;

  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r, S s) const noexcept -> Mat<D, 2, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  edge(Size i) const noexcept -> LineSegment<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isConvex() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool;
};

} // namespace um2

#include "Quadrilateral.inl"
