#pragma once

#include <um2/common/insertion_sort.hpp>
#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/quadrilateral.hpp>
#include <um2/geometry/modular_rays.hpp>

//==============================================================================
// QUADRATIC QUADRILATERAL
//==============================================================================

namespace um2
{

template <Int D>
class Polytope<2, 2, 8, D>
{
  static_assert(1 < D && D <= 3, "Only 2D, and 3D polygons are supported.");

public:
  using Vertex = Point<D>;
  using Edge = QuadraticSegment<D>;

private:
  Vertex _v[8];

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
  requires(sizeof...(Pts) == 8 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<8, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the surface of the polygon.
  // For quads: r in [0, 1], s in [0, 1]
  // F(r, s) -> (x, y, z)
  template <typename R, typename S>
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D>;

  // Get the i-th edge of the polygon.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int i) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  contains(Point2 const & p) const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  linearPolygon() const noexcept -> Quadrilateral<D>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  area() const noexcept -> Float
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  perimeter() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  centroid() const noexcept -> Point<D>
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>
  requires(D == 2);

  // If the polygon is counterclockwise oriented, returns true.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isCCW() const noexcept -> bool requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isConvex() const noexcept -> bool requires(D == 2);

  HOSTDEV constexpr void
  flip() noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 const & ray) const noexcept -> Vec<8, Float>
  requires(D == 2);

  // See the comments in the implementation for details.
  // meanChordLength has multiple definitions. Make sure you read the comments to
  // determine it's the one you want.
  PURE HOSTDEV [[nodiscard]] auto
  meanChordLength() const noexcept -> Float requires(D == 2);

}; // QuadraticQuadrilateral

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr QuadraticQuadrilateral<D>::Polytope(Vec<8, Vertex> const & v) noexcept
{
  for (Int i = 0; i < 8; ++i) {
    _v[i] = v[i];
  }
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::numEdges() noexcept -> Int
{
  return 4;
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 8);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::operator[](Int i) const noexcept -> Point<D> const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 8);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::vertices() const noexcept -> Point<D> const *
{
  return _v;
}

//==============================================================================
// interpolate
//==============================================================================

template <Int D>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::operator()(R const r, S const s) const noexcept -> Point<D>
{
  Float const xi = 2 * static_cast<Float>(r) - 1;
  Float const eta = 2 * static_cast<Float>(s) - 1;
  Float const w[8] = {(1 - xi) * (1 - eta) * (-xi - eta - 1) / 4,
                  (1 + xi) * (1 - eta) * (xi - eta - 1) / 4,
                  (1 + xi) * (1 + eta) * (xi + eta - 1) / 4,
                  (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4,
                  (1 - xi * xi) * (1 - eta) / 2,
                  (1 - eta * eta) * (1 + xi) / 2,
                  (1 - xi * xi) * (1 + eta) / 2,
                  (1 - eta * eta) * (1 - xi) / 2};
  return w[0] * _v[0] + w[1] * _v[1] +
         w[2] * _v[2] + w[3] * _v[3] +
         w[4] * _v[4] + w[5] * _v[5] +
         w[6] * _v[6] + w[7] * _v[7];
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::getEdge(Int i) const noexcept -> Edge
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 4);
  return (i < 3) ? QuadraticSegment<D>(_v[i], _v[i + 1], _v[i + 4])
                 : QuadraticSegment<D>(_v[3], _v[0], _v[7]);
}

//==============================================================================
// contains
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::contains(Point2 const & p) const noexcept -> bool requires(D == 2)
{
  for (Int i = 0; i < 4; ++i) {
    if (!getEdge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

//==============================================================================
// linearPolygon
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::linearPolygon() const noexcept -> Quadrilateral<D>
{
  return Quadrilateral<D>(_v[0], _v[1], _v[2], _v[3]);
}

//==============================================================================
// area
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::area() const noexcept -> Float
requires(D == 2)
{
  Float result = linearPolygon().area();
  result += enclosedArea(getEdge(0));
  result += enclosedArea(getEdge(1));
  result += enclosedArea(getEdge(2));
  result += enclosedArea(getEdge(3));
  return result;
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::perimeter() const noexcept -> Float
{
  Float result = getEdge(0).length();
  result += getEdge(1).length();
  result += getEdge(2).length();
  result += getEdge(3).length();
  return result;
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::centroid() const noexcept -> Point<D>
requires(D == 2)
{
  auto lin_poly = linearPolygon();
  Float area_sum = lin_poly.area();
  Point2 centroid_sum = area_sum * lin_poly.centroid();
  for (Int i = 0; i < 4; ++i) {
    auto const e = getEdge(i);
    Float const a = enclosedArea(e);
    area_sum += a;
    centroid_sum += a * enclosedCentroid(e);
  }
  return centroid_sum / area_sum;
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
requires(D == 2)
{
  AxisAlignedBox2 box = getEdge(0).boundingBox();
  box += getEdge(1).boundingBox();
  box += getEdge(2).boundingBox();
  box += getEdge(3).boundingBox();
  return box;
}

//==============================================================================
// isCCW
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::isCCW() const noexcept -> bool requires(D == 2)
{
  return linearPolygon().isCCW();
}

//==============================================================================
// isConvex
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::isConvex() const noexcept -> bool requires(D == 2)
{
  // If each edge is either straight, or curves left. 
  // AND the linear polygon polygon is convex.
  bool const lin_ok = linearPolygon().isConvex();
  if (!lin_ok) {
    return false;
  }
  auto const e0 = getEdge(0);
  auto const e1 = getEdge(1);
  auto const e2 = getEdge(2);
  auto const e3 = getEdge(3);
  bool const s_or_cl0 = isStraight(e0) || e0.curvesLeft(); 
  bool const s_or_cl1 = isStraight(e1) || e1.curvesLeft();
  bool const s_or_cl2 = isStraight(e2) || e2.curvesLeft();
  bool const s_or_cl3 = isStraight(e3) || e3.curvesLeft();
  bool const edges_ok = s_or_cl0 && s_or_cl1 && s_or_cl2 && s_or_cl3;
  return edges_ok;
}

//==============================================================================
// flip
//==============================================================================

template <Int D>
HOSTDEV constexpr void
QuadraticQuadrilateral<D>::flip() noexcept
{
  um2::swap(_v[1], _v[3]);
  um2::swap(_v[4], _v[7]);
}

//==============================================================================
// intersect
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D>::intersect(Ray2 const & ray) const noexcept -> Vec<8, Float>
requires(D == 2) {
  Vec<8, Float> result;
  for (Int i = 0; i < 4; ++i) {
    Vec2F const v = getEdge(i).intersect(ray);
    result[2 * i] = v[0];
    result[2 * i + 1] = v[1];
  }
  um2::insertionSort(result.begin(), result.end());
  return result;
}

//==============================================================================
// meanChordLength
//==============================================================================
// See the lengthy discussion in quadratic_triangle.hpp for details.

template <Int D>
PURE HOSTDEV auto
QuadraticQuadrilateral<D>::meanChordLength() const noexcept -> Float requires(D == 2)
{
  return um2::pi<Float> * area() / perimeter();
}

} // namespace um2
