#pragma once

#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/triangle.hpp>

//==============================================================================
// QUADRATIC TRIANGLE
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 2, 6, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  // NOLINTBEGIN(readability-identifier-naming)
  static constexpr Int N = 6; // Number of vertices
  // NOLINTEND(readability-identifier-naming)

  using Vertex = Point<D>;
  using Edge = QuadraticSegment<D>;

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

  HOSTDEV constexpr Polytope(Vec<N, Int> const & indices, Vertex const * vertices) noexcept
      : _v{vertices[indices[0]], vertices[indices[1]], 
           vertices[indices[2]], vertices[indices[3]],
           vertices[indices[4]], vertices[indices[5]]} 
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For triangles: r in [0, 1], s in [0, 1], constrained by r + s <= 1
  // F(r, s) -> R^D
  PURE HOSTDEV constexpr auto
  operator()(Float r, Float s) const noexcept -> Point<D>;

  // Jacobian of the interpolation function. 
  // [dF/dr, dF/ds]
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(Float r, Float s) const noexcept -> Mat<D, 2, Float>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  linearPolygon() const noexcept -> Triangle<D>;

  HOSTDEV constexpr void
  flip() noexcept;

  // 2D only
  //--------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox2
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  meanChordLength() const noexcept -> Float requires(D == 2);

  HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 ray, Float * buffer) const noexcept -> Int 
  requires(D == 2);

}; // QuadraticTriangle

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::operator()(Float const r, Float const s) const noexcept -> Point<D>
{
  Float const t = 1 - r - s;
  Float const w0 = t * (2 * t - 1);
  Float const w1 = r * (2 * r - 1);
  Float const w2 = s * (2 * s - 1);
  Float const w3 = 4 * r * t;
  Float const w4 = 4 * r * s;
  Float const w5 = 4 * s * t;
  return w0 * _v[0] + w1 * _v[1] + w2 * _v[2] + w3 * _v[3] + w4 * _v[4] + w5 * _v[5];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
jacobian(QuadraticTriangle<D> const & t6, Float const r, Float const s) noexcept
    -> Mat<D, 2, Float>
{
  Float const rr = (4 * r);
  Float const ss = (4 * s);
  Float const tt = rr + ss - 3;
  return Mat<D, 2, Float>{
    tt * (t6[0] - t6[3]) + (rr - 1) * (t6[1] - t6[3]) + ss * (t6[4] - t6[5]),
    tt * (t6[0] - t6[5]) + (ss - 1) * (t6[2] - t6[5]) + rr * (t6[4] - t6[3])};
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::jacobian(Float const r, Float const s) const noexcept -> Mat<D, 2, Float>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// getEdge
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::getEdge(Int i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// perimeter
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::perimeter() const noexcept -> Float
{
  return um2::perimeter(*this);
}

//==============================================================================
// linearPolygon
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::linearPolygon() const noexcept -> Triangle<D>
{
  return um2::linearPolygon(*this);
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
QuadraticTriangle<D>::flip() noexcept
{
  um2::swap(_v[1], _v[2]);
  um2::swap(_v[3], _v[5]);
}

//==============================================================================
// boundingBox
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::boundingBox() const noexcept -> AxisAlignedBox2
requires(D == 2)
{
  return um2::boundingBox(*this);
}

//==============================================================================
// area
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::area() const noexcept -> Float
requires(D == 2)
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::centroid() const noexcept -> Point<D>
requires(D == 2)
{
  return um2::centroid(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return linearPolygon().isCCW();
}

//==============================================================================
// contains
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::contains(Point2 p) const noexcept -> bool requires(D == 2)
{
  return um2::contains(*this, p);
}

//==============================================================================
// meanChordLength
//==============================================================================
// Defined in polygon.hpp

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  return um2::meanChordLength(*this);
}

//==============================================================================
// intersect
//==============================================================================
// Defined in polygon.hpp

template <Int D>
HOSTDEV constexpr auto
QuadraticTriangle<D>::intersect(Ray2 const ray, Float * const buffer) const noexcept -> Int
requires(D == 2)
{
  return um2::intersect(*this, ray, buffer);
}

} // namespace um2
