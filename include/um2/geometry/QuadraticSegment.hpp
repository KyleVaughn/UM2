#pragma once

#include <um2/math/Mat.hpp>

#include <um2/geometry/AxisAlignedBox.hpp>
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

  //  Q(r) = P₁ + rB + r²A,
  // where
  //  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  //  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // and
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  // NOTE: The equations above use 1-based indexing.

  Point<D, T> v[3];

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
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r) const noexcept -> Vec<D, T>;

  // Checks isApprox(v[2],  midpoint(v[0], v[1]))
  // NOTE: The segment may still be straight even if this returns false.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isStraight() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;
};

} // namespace um2

#include "QuadraticSegment.inl"