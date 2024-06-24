#pragma once

#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/quadrilateral.hpp>

//==============================================================================
// QUADRATIC QUADRILATERAL
//==============================================================================

namespace um2
{

template <Int D, class T>
class Polytope<2, 2, 8, D, T>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 8; // Number of vertices
  // NOLINTEND(readability-identifier-naming)

  using Vertex = Point<D, T>;
  using Edge = QuadraticSegment<D, T>;

private:
  Vertex _v[N];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  // Returns the i-th vertex of the polygon.
  PURE HOSTDEV constexpr auto
  operator[](Int i) noexcept -> Vertex &;

  // Returns the i-th vertex of the polygon.
  PURE HOSTDEV constexpr auto
  operator[](Int i) const noexcept -> Vertex const &;

  // Returns a pointer to the vertex array.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() const noexcept -> Vertex const *;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
    requires(sizeof...(Pts) == N && (std::same_as<Vertex, Pts> && ...))
  // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr Polytope(Vec<N, Int> const & indices,
                             Vertex const * vertices) noexcept
      : _v{vertices[indices[0]], vertices[indices[1]], vertices[indices[2]],
           vertices[indices[3]], vertices[indices[4]], vertices[indices[5]],
           vertices[indices[6]], vertices[indices[7]]}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For quads: r in [0, 1], s in [0, 1]
  // F(r, s) -> R^D
  PURE HOSTDEV constexpr auto
  operator()(T r, T s) const noexcept -> Point<D, T>;

  // Jacobian of the interpolation function.
  // [dF/dr, dF/ds]
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(T r, T s) const noexcept -> Mat<D, 2, T>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> T;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  linearPolygon() const noexcept -> Quadrilateral<D, T>;

  HOSTDEV constexpr void
  flip() noexcept;

  // 2D only
  //--------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D, T>
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2<T> p) const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] auto
  meanChordLength() const noexcept -> T
    requires(D == 2);

  HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2<T> ray, T * buffer) const noexcept -> Int
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  hasSelfIntersection() const noexcept -> bool
    requires(D == 2);

  HOSTDEV [[nodiscard]] constexpr auto
  hasSelfIntersection(Point2<T> * buffer) const noexcept -> bool
    requires(D == 2);

}; // QuadraticQuadrilateral

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator[](Int i) const noexcept -> Point<D, T> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::vertices() const noexcept -> Point<D, T> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator()(T const r,
                                      T const s) const noexcept -> Point<D, T>
{
  auto constexpr hf = castIfNot<T>(0.5);
  auto constexpr qtr = castIfNot<T>(0.25);
  T const xi = 2 * r - 1;
  T const eta = 2 * s - 1;
  T const w0 = qtr * (1 - xi) * (1 - eta) * (-xi - eta - 1);
  T const w1 = qtr * (1 + xi) * (1 - eta) * (xi - eta - 1);
  T const w2 = qtr * (1 + xi) * (1 + eta) * (xi + eta - 1);
  T const w3 = qtr * (1 - xi) * (1 + eta) * (-xi + eta - 1);
  T const w4 = hf * (1 - xi * xi) * (1 - eta);
  T const w5 = hf * (1 - eta * eta) * (1 + xi);
  T const w6 = hf * (1 - xi * xi) * (1 + eta);
  T const w7 = hf * (1 - eta * eta) * (1 - xi);
  return w0 * _v[0] + w1 * _v[1] + w2 * _v[2] + w3 * _v[3] + w4 * _v[4] + w5 * _v[5] +
         w6 * _v[6] + w7 * _v[7];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
jacobian(QuadraticQuadrilateral<D, T> const & q, T const r,
         T const s) noexcept -> Mat<D, 2, T>
{
  T const xi = 2 * r - 1;
  T const eta = 2 * s - 1;
  T const xi_eta = xi * eta;
  T const xi_xi = xi * xi;
  T const eta_eta = eta * eta;
  T const w0 = (eta - eta_eta) / 2;
  T const w1 = (eta + eta_eta) / 2;
  T const w2 = (xi - xi_eta);
  T const w3 = (xi + xi_eta);
  T const w4 = 1 - eta_eta;
  T const w5 = (xi - xi_xi) / 2;
  T const w6 = (xi + xi_xi) / 2;
  T const w7 = eta - xi_eta;
  T const w8 = eta + xi_eta;
  T const w9 = 1 - xi_xi;
  return Mat<D, 2, T>(
      w0 * (q[0] - q[1]) + w1 * (q[2] - q[3]) + w2 * (q[0] + q[1] - 2 * q[4]) +
          w3 * (q[2] + q[3] - 2 * q[6]) + w4 * (q[5] - q[7]),
      w5 * (q[0] - q[3]) + w6 * (q[2] - q[1]) + w7 * (q[0] + q[3] - 2 * q[7]) +
          w8 * (q[1] + q[2] - 2 * q[5]) + w9 * (q[6] - q[4]));
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::jacobian(T const r,
                                    T const s) const noexcept -> Mat<D, 2, T>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// getEdge
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::getEdge(Int i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// perimeter
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::perimeter() const noexcept -> T
{
  return um2::perimeter(*this);
}

//==============================================================================
// linearPolygon
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::linearPolygon() const noexcept -> Quadrilateral<D, T>
{
  return um2::linearPolygon(*this);
}

//==============================================================================
// flip
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr void
QuadraticQuadrilateral<D, T>::flip() noexcept
{
  um2::swap(_v[1], _v[3]);
  um2::swap(_v[4], _v[7]);
}

//==============================================================================
// boundingBox
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
  requires(D == 2)
{
  return um2::boundingBox(*this);
}

//==============================================================================
// area
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::area() const noexcept -> T
  requires(D == 2)
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::centroid() const noexcept -> Point<D, T>
  requires(D == 2)
{
  return um2::centroid(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::isCCW() const noexcept -> bool
  requires(D == 2)
{
  return linearPolygon().isCCW();
}

//==============================================================================
// contains
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::contains(Point2<T> p) const noexcept -> bool
  requires(D == 2)
{
  return um2::contains(*this, p);
}

//==============================================================================
// meanChordLength
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV auto
QuadraticQuadrilateral<D, T>::meanChordLength() const noexcept -> T
  requires(D == 2)
{
  return um2::meanChordLength(*this);
}

//==============================================================================
// intersect
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::intersect(Ray2<T> const ray,
                                     T * const buffer) const noexcept -> Int
  requires(D == 2)
{
  return um2::intersect(*this, ray, buffer);
}

//==============================================================================
// hasSelfIntersection
//==============================================================================
// Defined in polygon.hpp

template <Int D, class T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::hasSelfIntersection() const noexcept -> bool
  requires(D == 2)
{
  return um2::hasSelfIntersection(*this);
}

template <Int D, class T>
HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::hasSelfIntersection(Point2<T> * const buffer) const noexcept
    -> bool
  requires(D == 2)
{
  return um2::hasSelfIntersection(*this, buffer);
}

} // namespace um2
