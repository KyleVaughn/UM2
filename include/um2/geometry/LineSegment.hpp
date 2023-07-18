#pragma once

#include <um2/math/Mat.hpp>

#include <um2/geometry/AxisAlignedBox.hpp>
#include <um2/geometry/Polytope.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// LINE SEGMENT
// -----------------------------------------------------------------------------
// A 1-polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 2 vertices are D-dimensional points of type T.

template <typename T>
using LineSegment2 = LineSegment<2, T>;
using LineSegment2f = LineSegment2<float>;
using LineSegment2d = LineSegment2<double>;

template <Size D, typename T>
struct Polytope<1, 1, 2, D, T> {

  Point<D, T> v[2];

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Point<D, T> &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Point<D, T> const &;

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Polytope() noexcept = default;

  HOSTDEV constexpr Polytope(Point<D, T> const & p0, Point<D, T> const & p1) noexcept;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Point<D, T>;

  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(R /*r*/) const noexcept -> Vec<D, T>;

  // Get the rotation matrix that transforms the line segment such that it is
  // aligned with the x-axis.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat<D, D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Point<D, T> const & p) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Point<D, T> const & p) const noexcept -> T;
};

} // namespace um2

#include "LineSegment.inl"
