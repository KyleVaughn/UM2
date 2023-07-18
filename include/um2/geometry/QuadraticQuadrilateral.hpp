#pragma once

#include <um2/geometry/QuadraticSegment.hpp>
#include <um2/geometry/Quadrilateral.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// QUADRATIC QUADRILATERAL 
// -----------------------------------------------------------------------------
// A 2-polytope, of polynomial order 1, represented by the connectivity
// of its vertices. These 4 vertices are D-dimensional points of type T.

template <typename T>
using QuadraticQuadrilateral2 = QuadraticQuadrilateral<2, T>;
using QuadraticQuadrilateral2f = QuadraticQuadrilateral2<float>;
using QuadraticQuadrilateral2d = QuadraticQuadrilateral2<double>;

template <typename T>
using QuadraticQuadrilateral3 = QuadraticQuadrilateral<3, T>;
using QuadraticQuadrilateral3f = QuadraticQuadrilateral3<float>;
using QuadraticQuadrilateral3d = QuadraticQuadrilateral3<double>;

template <Size D, typename T>
struct Polytope<2, 2, 8, D, T> {

  Point<D, T> v[8];

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
  PURE HOSTDEV [[nodiscard]] constexpr auto jacobian(R r, S s) const noexcept
      -> Mat<D, 2, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  edge(Size i) const noexcept -> QuadraticSegment<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point<D, T> const & p) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  linearPolygon() const noexcept -> Quadrilateral<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;
};

} // namespace um2

#include "QuadraticQuadrilateral.inl"
