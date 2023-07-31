#pragma once

#include <um2/geometry/QuadraticSegment.hpp>
#include <um2/geometry/Triangle.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// QUADRATIC TRIANGLE
// -----------------------------------------------------------------------------
// A 2-polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 3 vertices are D-dimensional points of type T.

template <typename T>
using QuadraticTriangle2 = QuadraticTriangle<2, T>;
using QuadraticTriangle2f = QuadraticTriangle2<float>;
using QuadraticTriangle2d = QuadraticTriangle2<double>;

template <typename T>
using QuadraticTriangle3 = QuadraticTriangle<3, T>;
using QuadraticTriangle3f = QuadraticTriangle3<float>;
using QuadraticTriangle3d = QuadraticTriangle3<double>;

template <Size D, typename T>
struct Polytope<2, 2, 6, D, T> {

  Point<D, T> v[6];

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Polytope() noexcept = default;

  HOSTDEV constexpr Polytope(Point<D, T> const & p0, Point<D, T> const & p1,
                             Point<D, T> const & p2, Point<D, T> const & p3,
                             Point<D, T> const & p4, Point<D, T> const & p5) noexcept;

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
  PURE HOSTDEV [[nodiscard]] constexpr auto
  operator()(R r, S s) const noexcept -> Point<D, T>;

  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r, S s) const noexcept -> Mat<D, 2, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  edge(Size i) const noexcept -> QuadraticSegment<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  linearPolygon() const noexcept -> Triangle<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;
};

} // namespace um2

#include "QuadraticTriangle.inl"
