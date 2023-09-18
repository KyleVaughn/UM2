#pragma once

#include <um2/geometry/Polytope.hpp>
#include <um2/geometry/Ray.hpp>

// Need complex for quadratic segments
#if UM2_USE_CUDA
#  include <cuda/std/complex>
#else
#  include <complex>
#endif

#include <iostream>

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
  // NOLINTNEXTLINE(google-explicit-constructor) justified: implicit conversion desired
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : v{args...}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate the polytope at the given parameter value.
  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Point<D, T>;

  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r) const noexcept -> Vec<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat<D, D, T>
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Point<D, T> const & p) const noexcept -> bool
    requires(D == 2);

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

//==============================================================================
// interpolate
//==============================================================================

template <Size P, Size N, Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
interpolate(Dion<P, N, D, T> const & dion, R r) noexcept -> Point<D, T>;

//==============================================================================
// jacobian
//==============================================================================

template <Size P, Size N, Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
jacobian(Dion<P, N, D, T> const & dion, R r) noexcept -> Point<D, T>;

//==============================================================================
// getRotation
//==============================================================================

template <Size P, Size N, typename T>
PURE HOSTDEV constexpr auto
getRotation(PlanarDion<P, N, T> const & dion) noexcept -> Mat2x2<T>;

//==============================================================================
// pointIsLeft
//==============================================================================

template <Size P, Size N, typename T>
PURE HOSTDEV constexpr auto
pointIsLeft(PlanarDion<P, N, T> const & dion, Point2<T> const & p) noexcept -> bool;

//==============================================================================
// length
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
length(Dion<P, N, D, T> const & dion) noexcept -> T;

//==============================================================================
// boundingBox
//==============================================================================

// Defined in Polytope.hpp for the line segment, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(QuadraticSegment<D, T> const & q) noexcept -> AxisAlignedBox<D, T>;

//==============================================================================
// pointClosestTo
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
pointClosestTo(Dion<P, N, D, T> const & dion, Point<D, T> const & p) noexcept -> T;

//==============================================================================
// isStraight
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D, T> const & q) noexcept -> bool;

//==============================================================================
// getBezierControlPoint
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
getBezierControlPoint(QuadraticSegment<D, T> const & q) noexcept -> Point<D, T>;

//==============================================================================
// enclosedArea
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
enclosedArea(QuadraticSegment2<T> const & q) noexcept -> T;

//==============================================================================
// enclosedCentroid
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
enclosedCentroid(QuadraticSegment2<T> const & q) noexcept -> Point2<T>;

//==============================================================================
// intersect
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
intersect(LineSegment2<T> const & line, Ray2<T> const & ray) noexcept -> T;

template <typename T>
PURE HOSTDEV constexpr auto
intersect(QuadraticSegment2<T> const & q, Ray2<T> const & ray) noexcept -> Vec2<T>;

template <typename T>
PURE HOSTDEV auto
intersect(AxisAlignedBox2<T> const & box, Ray2<T> const & ray) noexcept -> Vec2<T>;

} // namespace um2

#include "Dion.inl"
