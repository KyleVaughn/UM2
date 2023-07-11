#pragma once

#include <um2/geometry/Polytope.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// QUADRATIC SEGMENT
// -----------------------------------------------------------------------------
// A 1-dimensional polytope, of polynomial order 2, represented by the connectivity
// of its vertices. These 3 vertices are D-dimensional points of type T.

template <typename T>
using QuadraticSegment2 = QuadraticSegment<2, T>;
using QuadraticSegment2f = QuadraticSegment2<float>;
using QuadraticSegment2d = QuadraticSegment2<double>;

template <Size D, typename T>
struct Polytope<1, 2, 3, D, T> {

  Point<D, T> vertices[3];

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

  HOSTDEV constexpr Polytope(Point<D, T> const & p0, Point<D, T> const & p1,
                             Point<D, T> const & p2) noexcept;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Point<D, T>;

  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(R /*r*/) const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;
};

} // namespace um2

#include "QuadraticSegment.inl"
