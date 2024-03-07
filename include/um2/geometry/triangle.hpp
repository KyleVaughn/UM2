#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// TRIANGLE
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 1, 3, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  using Vertex = Point<D>;
  using Edge = LineSegment<D>;

private:
  Vertex _v[3];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  // Returns the number of edges in the polygon.
  PURE HOSTDEV static constexpr auto
  numEdges() noexcept -> Int;

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
  requires(sizeof...(Pts) == 3 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<3, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For triangles: r in [0, 1], s in [0, 1], constrained by r + s <= 1
  // F(r, s) -> (x, y, z)
  template <typename R, typename S>
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D>;

  // J(r, s) -> [dF/dr, dF/ds]
  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r, S s) const noexcept -> Mat<D, 2, Float>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] static constexpr auto
  isConvex() noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool requires(D == 2);

  HOSTDEV constexpr void
  flip() noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 ray) const noexcept -> Vec<3, Float>
  requires(D == 2);

  // See the comments in the implementation for details.
  // meanChordLength has multiple definitions. Make sure you read the comments to
  // determine it's the one you want.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  meanChordLength() const noexcept -> Float requires(D == 2);

}; // Triangle

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr Triangle<D>::Polytope(Vec<3, Vertex> const & v) noexcept
{
  _v[0] = v[0];
  _v[1] = v[1];
  _v[2] = v[2];
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::numEdges() noexcept -> Int
{
  return 3;
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 3);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 3);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Triangle<D>::operator()(R const r, S const s) const noexcept -> Point<D>
{
  // T(r, s) = (1 - r - s) v0 + r v1 + s v2
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const w0 = 1 - rr - ss;
  // Float const w1 = rr;
  // Float const w2 = ss;
  return w0 * _v[0] + rr * _v[1] + ss * _v[2];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Triangle<D>::jacobian(R /*r*/, S /*s*/) const noexcept -> Mat<D, 2, Float>
{
  return Mat<D, 2, Float>(_v[1] - _v[0], _v[2] - _v[0]);
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::getEdge(Int i) const noexcept -> Edge
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 3);
  return (i < 2) ? Edge(_v[i], _v[i + 1]) : Edge(_v[2], _v[0]);
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::isConvex() noexcept -> bool
{
  return true;
}

//==============================================================================
// contains
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  Vec2F const a = _v[1] - _v[0];
  Vec2F const b = _v[2] - _v[0];
  Vec2F const c = p - _v[0];
  Float const invdet_ab = 1 / a.cross(b);
  Float const r = c.cross(b) * invdet_ab;
  Float const s = a.cross(c) * invdet_ab;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
}

//==============================================================================
// area
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::area() const noexcept -> Float
{
  Vec<D, Float> const v10 = _v[1] - _v[0];
  Vec<D, Float> const v20 = _v[2] - _v[0];
  if constexpr (D == 2) {
    return v10.cross(v20) / 2; // this is the signed area
  } else {
    return v10.cross(v20).norm() / 2; // this is the unsigned area
  }
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::perimeter() const noexcept -> Float
{
  return _v[0].distanceTo(_v[1]) + _v[1].distanceTo(_v[2]) + _v[2].distanceTo(_v[0]);
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::centroid() const noexcept -> Point<D>
{
  return (_v[0] + _v[1] + _v[2]) / 3;
}

//==============================================================================
// boundingBox
//==============================================================================

// Defined in Polytope.hpp for linear polygons, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return areCCW(_v[0], _v[1], _v[2]);
}

//==============================================================================
// intersect
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::intersect(Ray2 const ray) const noexcept -> Vec3F
requires(D == 2) {
  Vec3F result;
  result[0] = Edge(_v[0], _v[1]).intersect(ray);
  result[1] = Edge(_v[1], _v[2]).intersect(ray);
  result[2] = Edge(_v[2], _v[0]).intersect(ray);
  return result;
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
Triangle<D>::flip() noexcept
{
  um2::swap(_v[1], _v[2]);
}

//==============================================================================
// meanChordLength
//==============================================================================
// For a convex planar polygon, the mean chord length is simply pi * area / perimeter.
// De Kruijf, W. J. M., and J. L. Kloosterman.
// "On the average chord length in reactor physics." Annals of Nuclear Energy 30.5 (2003):
// 549-553.

template <Int D>
PURE HOSTDEV constexpr auto
Triangle<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  return um2::pi<Float> * area() / perimeter();
}

} // namespace um2
