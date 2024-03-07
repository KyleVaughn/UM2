#pragma once

#include <um2/geometry/line_segment.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// QUADRILATERAL
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 1, 4, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  using Vertex = Point<D>;
  using Edge = LineSegment<D>;

private:
  Vertex _v[4];

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
  requires(sizeof...(Pts) == 4 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<4, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For quads: r in [0, 1], s in [0, 1]
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

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isConvex() const noexcept -> bool
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isApproxConvex() const noexcept -> bool
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float
  requires(D == 2);

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
  intersect(Ray2 ray) const noexcept -> Vec<4, Float>
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
HOSTDEV constexpr Quadrilateral<D>::Polytope(Vec<4, Vertex> const & v) noexcept
{
  _v[0] = v[0];
  _v[1] = v[1];
  _v[2] = v[2];
  _v[3] = v[3];
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::numEdges() noexcept -> Int
{
  return 4;
}

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 4);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 4);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::operator()(R const r, S const s) const noexcept -> Point<D>
{
  // Q(r, s) =
  // (1 - r) (1 - s) v0 +
  // (    r) (1 - s) v1 +
  // (    r) (    s) v2 +
  // (1 - r) (    s) v3
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const w0 = (1 - rr) * (1 - ss);
  Float const w1 = rr * (1 - ss);
  Float const w2 = rr * ss;
  Float const w3 = (1 - rr) * ss;
  return w0 * _v[0] + w1 * _v[1] + w2 * _v[2] + w3 * _v[3];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::jacobian(R r, S s) const noexcept -> Mat<D, 2, Float>
{
  // TODO(kcvaughn): Is this correct, or is it transposed?
  auto const rr = static_cast<Float>(r);
  auto const ss = static_cast<Float>(s);
  Float const w0 = 1 - ss;
  // Float const w1 = ss;
  Float const w2 = 1 - rr;
  // Float const w3 = rr;
  return Mat<D, 2, Float>(
    w0 * (_v[1] - _v[0]) - ss * (_v[3] - _v[2]),
    w2 * (_v[3] - _v[0]) - rr * (_v[1] - _v[2]));
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::getEdge(Int i) const noexcept -> Edge
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 4);
  return (i < 3) ? Edge(_v[i], _v[i + 1]) : Edge(_v[3], _v[0]);
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::isConvex() const noexcept -> bool
requires(D == 2)
{
  // Benchmarking shows it is faster to compute the areCCW() test for each
  // edge, then return based on the AND of the results, rather than compute
  // the areCCW one at a time and return as soon as one is false.
  bool const b0 = areCCW(_v[0], _v[1], _v[2]);
  bool const b1 = areCCW(_v[1], _v[2], _v[3]);
  bool const b2 = areCCW(_v[2], _v[3], _v[0]);
  bool const b3 = areCCW(_v[3], _v[0], _v[1]);
  return b0 && b1 && b2 && b3;
}

//==============================================================================
// isApproxConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::isApproxConvex() const noexcept -> bool
requires(D == 2)
{
  bool const b0 = areApproxCCW(_v[0], _v[1], _v[2]);
  bool const b1 = areApproxCCW(_v[1], _v[2], _v[3]);
  bool const b2 = areApproxCCW(_v[2], _v[3], _v[0]);
  bool const b3 = areApproxCCW(_v[3], _v[0], _v[1]);
  return b0 && b1 && b2 && b3;
}

//==============================================================================
// contains
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  ASSERT(isApproxConvex());
  bool const b0 = areCCW(_v[0], _v[1], p);
  bool const b1 = areCCW(_v[1], _v[2], p);
  bool const b2 = areCCW(_v[2], _v[3], p);
  bool const b3 = areCCW(_v[3], _v[0], p);
  return b0 && b1 && b2 && b3;
}

//==============================================================================
// area
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::area() const noexcept -> Float
requires(D == 2)
{
  ASSERT(isApproxConvex());
  // (v2 - v0).cross(v3 - v1) / 2
  Vec2F const v20 = _v[2] - _v[0];
  Vec2F const v31 = _v[3] - _v[1];
  return v20.cross(v31) / 2;
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::perimeter() const noexcept -> Float
{
  Float const d01 = _v[0].distanceTo(_v[1]);
  Float const d12 = _v[1].distanceTo(_v[2]);
  Float const d23 = _v[2].distanceTo(_v[3]);
  Float const d30 = _v[3].distanceTo(_v[0]);
  return d01 + d12 + d23 + d30;
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::centroid() const noexcept -> Point<D>
{
  // Algorithm: Decompose the quadrilateral into two triangles and
  // compute the centroid of each triangle. The centroid of the
  // quadrilateral is the weighted average of the centroids of the
  // two triangles, where the weights are the areas of the triangles.
  ASSERT(isApproxConvex());
  // If the quadrilateral is not convex, then we need to choose the correct
  // two triangles to decompose the quadrilateral into. If the quadrilateral
  // is convex, any two triangles will do.
  Vec2F const v10 = _v[1] - _v[0];
  Vec2F const v20 = _v[2] - _v[0];
  Vec2F const v30 = _v[3] - _v[0];
  // Compute the area of each triangle
  Float const a1 = v10.cross(v20);
  Float const a2 = v20.cross(v30);
  Float const a12 = a1 + a2;
  // Compute the centroid of each triangle
  // (v0 + v1 + v2) / 3
  // Each triangle shares v0 and v2, so we factor out the common terms
  return (a1 * _v[1] + a2 * _v[3] + a12 * (_v[0] + _v[2])) / (3 * a12);
}

//==============================================================================
// boundingBox
//==============================================================================

// Defined in Polytope.hpp for linear polygons, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  bool const b0 = areCCW(_v[0], _v[1], _v[2]);
  bool const b1 = areCCW(_v[0], _v[2], _v[3]);
  return b0 && b1;
}

//==============================================================================
// intersect
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
Quadrilateral<D>::intersect(Ray2 const ray) const noexcept -> Vec<4, Float>
requires(D == 2) {
  Vec<4, Float> result;
  result[0] = Edge(_v[0], _v[1]).intersect(ray);
  result[1] = Edge(_v[1], _v[2]).intersect(ray);
  result[2] = Edge(_v[2], _v[3]).intersect(ray);
  result[3] = Edge(_v[3], _v[0]).intersect(ray);
  return result;
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
Quadrilateral<D>::flip() noexcept
{
  um2::swap(_v[1], _v[3]);
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
Quadrilateral<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  ASSERT(isApproxConvex());
  return um2::pi<Float> * area() / perimeter();
}

} // namespace um2
