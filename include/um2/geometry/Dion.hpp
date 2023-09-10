#pragma once

#include <um2/geometry/Polytope.hpp>
#include <um2/geometry/Ray.hpp>

// Need complex for quadratic segments
#if UM2_USE_CUDA
#  include <cuda/std/complex>
#else
#  include <complex>
#endif

//==============================================================================
// DION
//==============================================================================
//
// A 1-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points of type T.
//
// For Dions:
//   LineSegment (P = 1, N = 2)
//   QuadraticSegment (P = 2, N = 3)

// For quadratic segments, the parametric equation is
//  Q(r) = P₁ + rB + r²A,
// where
//  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
//  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
// and
// V₁₃ = q[3] - q[1]
// V₂₃ = q[3] - q[2]
// NOTE: The equations above use 1-based indexing.

namespace um2
{

template <Size P, Size N, Size D, typename T>
struct Polytope<1, P, N, D, T> {

  Point<D, T> v[N];

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Point<D, T> &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Point<D, T> const &;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
    requires(sizeof...(Pts) == N && (std::same_as<Point<D, T>, Pts> && ...))
  // NOLINTBEGIN(google-explicit-constructor) justification: implicit conversion
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : v{args...}
  {
  }
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Methods
  //==============================================================================

  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Point<D, T>;

  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r) const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat<D, D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  pointClosestTo(Point<D, T> const & p) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Point<D, T> const & p) const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Point<D, T> const & p) const noexcept -> T;

}; // Dion

} // namespace um2

#include "Dion.inl"
