#pragma once

#include <um2/geometry/Polytope.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// LINE SEGMENT
// -----------------------------------------------------------------------------

template <typename T>
using LineSegment2 = LineSegment<2, T>;
using LineSegment2f = LineSegment2<float>;
using LineSegment2d = LineSegment2<double>;

template <Size D, typename T>
struct Polytope<1, 1, 2, D, T> {

  // L(r) = v0 + r * (v1 - v0) 
  //      = w0 + r * w1
  Point<D, T> w[2];

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Polytope() noexcept = default;

  HOSTDEV constexpr Polytope(Point<D, T> const & p0, Point<D, T> const & p1) noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV constexpr auto
  getVertex(Size i) const noexcept -> Point<D, T>;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Point<D, T>;

  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(R /*r*/) const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;
};

} // namespace um2

#include "LineSegment.inl"
