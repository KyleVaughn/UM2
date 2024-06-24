#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/geometry/polygon.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// QUADRILATERAL
//==============================================================================

namespace um2
{

template <Int D, class T>
class Polytope<2, 1, 4, D, T>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 4; // Number of vertices
  // NOLINTEND(readability-identifier-naming)

  using Vertex = Point<D, T>;
  using Edge = LineSegment<D, T>;

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
           vertices[indices[3]]}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For quads: r in [0, 1], s in [0, 1]
  // F(r, s) -> (x, y, z)
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
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  // 2D only
  //--------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isConvex() const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isApproxConvex() const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> T
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point2<T>
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool
    requires(D == 2);

  HOSTDEV constexpr void
  flip() noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2<T> const & p) const noexcept -> bool
    requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  meanChordLength() const noexcept -> T
    requires(D == 2);

  HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2<T> ray, T * buffer) const noexcept -> Int
    requires(D == 2);

}; // Quadrilateral

//==============================================================================
// Accessors
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator[](Int i) const noexcept -> Point<D, T> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::vertices() const noexcept -> Point<D, T> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator()(T const r, T const s) const noexcept -> Point<D, T>
{
  // Q(r, s) =
  // (1 - r) (1 - s) v0 +
  // (    r) (1 - s) v1 +
  // (    r) (    s) v2 +
  // (1 - r) (    s) v3
  T const w0 = (1 - r) * (1 - s);
  T const w1 = r * (1 - s);
  T const w2 = r * s;
  T const w3 = (1 - r) * s;
  return w0 * _v[0] + w1 * _v[1] + w2 * _v[2] + w3 * _v[3];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
jacobian(Quadrilateral<D, T> const & q, T const r,
         T const s) noexcept -> Mat<D, 2, T>
{
  // jac.col(0) = w0 * (v1 - v0) - s (v3 - v2)
  // jac.col(1) = w2 * (v3 - v0) - r (v1 - v2)
  T const w0 = 1 - s;
  T const w2 = 1 - r;
  return Mat<D, 2, T>(w0 * (q[1] - q[0]) - s * (q[3] - q[2]),
                          w2 * (q[3] - q[0]) - r * (q[1] - q[2]));
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::jacobian(T const r,
                           T const s) const noexcept -> Mat<D, 2, T>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// getEdge
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all linear polygons.

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::getEdge(Int i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// perimeter
//==============================================================================
// Defined in polygon.hpp, since the procedure is the same for all linear polygons

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::perimeter() const noexcept -> T
{
  return um2::perimeter(*this);
}

//==============================================================================
// boundingBox
//==============================================================================
// Defined in polytope.hpp, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isConvex
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
isConvex(Quadrilateral2<T> const & q) noexcept -> bool
{
  // Benchmarking shows it is faster to compute the areCCW() test for each
  // edge, then return based on the AND of the results, rather than compute
  // the areCCW one at a time and return as soon as one is false.
  bool const b0 = areCCW(q[0], q[1], q[2]);
  bool const b1 = areCCW(q[1], q[2], q[3]);
  bool const b2 = areCCW(q[2], q[3], q[0]);
  bool const b3 = areCCW(q[3], q[0], q[1]);
  return b0 && b1 && b2 && b3;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::isConvex() const noexcept -> bool
  requires(D == 2)
{
  return um2::isConvex(*this);
}

template <class T>
PURE HOSTDEV constexpr auto
isApproxConvex(Quadrilateral2<T> const & q) noexcept -> bool
{
  bool const b0 = areApproxCCW(q[0], q[1], q[2]);
  bool const b1 = areApproxCCW(q[1], q[2], q[3]);
  bool const b2 = areApproxCCW(q[2], q[3], q[0]);
  bool const b3 = areApproxCCW(q[3], q[0], q[1]);
  return b0 && b1 && b2 && b3;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::isApproxConvex() const noexcept -> bool
  requires(D == 2)
{
  return um2::isApproxConvex(*this);
}

//==============================================================================
// area
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
area(Quadrilateral2<T> const & q) noexcept -> T
{
  ASSERT(isApproxConvex(q));
  // (v2 - v0).cross(v3 - v1) / 2
  Vec2<T> const v20 = q[2] - q[0];
  Vec2<T> const v31 = q[3] - q[1];
  return v20.cross(v31) / 2;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::area() const noexcept -> T
  requires(D == 2)
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
centroid(Quadrilateral2<T> const & q) noexcept -> Point2<T>
{
  // Algorithm: Decompose the quadrilateral into two triangles and
  // compute the centroid of each triangle. The centroid of the
  // quadrilateral is the weighted average of the centroids of the
  // two triangles, where the weights are the areas of the triangles.
  ASSERT(isApproxConvex(q));
  // If the quadrilateral is not convex, then we need to choose the correct
  // two triangles to decompose the quadrilateral into. If the quadrilateral
  // is convex, any two triangles will do.
  Vec2<T> const v10 = q[1] - q[0];
  Vec2<T> const v20 = q[2] - q[0];
  Vec2<T> const v30 = q[3] - q[0];
  // Compute the area of each triangle
  T const a1 = v10.cross(v20);
  T const a2 = v20.cross(v30);
  T const a12 = a1 + a2;
  // Compute the centroid of each triangle
  // (v0 + v1 + v2) / 3
  // Each triangle shares v0 and v2, so we factor out the common terms
  return (a1 * q[1] + a2 * q[3] + a12 * (q[0] + q[2])) / (3 * a12);
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::centroid() const noexcept -> Point2<T>
  requires(D == 2)
{
  return um2::centroid(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
isCCW(Quadrilateral2<T> const & q) noexcept -> bool
{
  bool const b0 = areApproxCCW(q[0], q[1], q[2]);
  bool const b1 = areApproxCCW(q[0], q[2], q[3]);
  return b0 && b1;
}

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::isCCW() const noexcept -> bool
  requires(D == 2)
{
  return um2::isCCW(*this);
}

//==============================================================================
// flip
//==============================================================================

template <Int D, class T>
HOSTDEV constexpr void
Quadrilateral<D, T>::flip() noexcept
{
  um2::swap(_v[1], _v[3]);
}

//==============================================================================
// contains
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::contains(Point2<T> const & p) const noexcept -> bool
  requires(D == 2)
{
  ASSERT(isApproxConvex());
  bool const b0 = areCCW(_v[0], _v[1], p);
  bool const b1 = areCCW(_v[1], _v[2], p);
  bool const b2 = areCCW(_v[2], _v[3], p);
  bool const b3 = areCCW(_v[3], _v[0], p);
  return b0 && b1 && b2 && b3;
}

//==============================================================================
// meanChordLength
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::meanChordLength() const noexcept -> T
  requires(D == 2)
{
  return um2::meanChordLength(*this);
}

//==============================================================================
// intersect
//==============================================================================

template <Int D, class T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::intersect(Ray2<T> const ray, T * buffer) const noexcept -> Int
  requires(D == 2)
{
  return um2::intersect(*this, ray, buffer);
}

} // namespace um2
