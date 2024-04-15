#pragma once

#include <um2/geometry/polytope.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// POLYGON
//==============================================================================
// Functions that are common to all polygons:
//  - getEdge
//  - perimeter
//  - boundingBox
//  - area
//  - centroid
//
// Functions that are specific to planar polygons:
//  - contains(Point2)
//  - meanChordLength
//  - intersect(Ray2)
//  - hasSelfIntersection (quadratic polygons only)

#define STATIC_ASSERT_VALID_POLYGON \
  static_assert(D >= 2, "Polygons must be embedded in at least 2 dimensions"); \
  static_assert(N >= 2, "A polygon must have at least 2 vertices");

namespace um2
{

//==============================================================================
// polygonNumEdges
//==============================================================================

template <Int P, Int N>
CONST HOSTDEV [[nodiscard]] constexpr auto
polygonNumEdges() noexcept -> Int
{
  return N / P;
}

//==============================================================================
// linearPolygon
//==============================================================================

template <Int N, Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
linearPolygon(QuadraticPolygon<N, D> const & p) noexcept -> LinearPolygon<N / 2, D>
{
  LinearPolygon<N / 2, D> result;
  for (Int i = 0; i < N / 2; ++i) {
    result[i] = p[i];
  }
  return result; 
}

//==============================================================================
// getEdge
//==============================================================================
// Return the i-th edge of the polygon.

template <Int N, Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
getEdge(LinearPolygon<N, D> const & lp, Int const i) noexcept -> LineSegment<D>
{
  STATIC_ASSERT_VALID_POLYGON;
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return (i < N - 1) ? LineSegment<D>(lp[i], lp[i + 1])
                     : LineSegment<D>(lp[N - 1], lp[0]);
}

template <Int N, Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
getEdge(QuadraticPolygon<N, D> const & qp, Int const i) noexcept -> QuadraticSegment<D>
{
  STATIC_ASSERT_VALID_POLYGON;
  Int constexpr m = polygonNumEdges<2, N>();
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < m);
  return (i < m - 1) ? QuadraticSegment<D>(qp[i], qp[i + 1], qp[i + m])
                     : QuadraticSegment<D>(qp[m - 1], qp[0], qp[N - 1]);
}

//==============================================================================
// perimeter
//==============================================================================
// The perimeter of a polygon is the sum of th lengths of its edges.
// For a linear polygon, we can simply sum the distances between consecutive
// vertices.

template <Int N, Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
perimeter(LinearPolygon<N, D> const & p) noexcept -> Float
{
  STATIC_ASSERT_VALID_POLYGON;
  // Take care of the last edge (wraparound) separately.
  Float result = p[N - 1].distanceTo(p[0]);
  for (Int i = 0; i < N - 1; ++i) {
    result += p[i].distanceTo(p[i + 1]);
  }
  ASSERT(result > 0);
  return result;
}

template <Int N, Int D>
PURE HOSTDEV [[nodiscard]] constexpr auto
perimeter(QuadraticPolygon<N, D> const & p) noexcept -> Float
{
  STATIC_ASSERT_VALID_POLYGON;
  Int constexpr m = polygonNumEdges<2, N>();
  Float result = p.getEdge(0).length();
  for (Int i = 1; i < m; ++i) {
    result += p.getEdge(i).length();
  }
  ASSERT(result > 0);
  return result;
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
boundingBox(PlanarQuadraticPolygon<N> const & p) noexcept -> AxisAlignedBox2
{
  Int constexpr m = polygonNumEdges<2, N>();
  AxisAlignedBox2 result = p.getEdge(0).boundingBox();
  for (Int i = 1; i < m; ++i) {
    result += p.getEdge(i).boundingBox();
  }
  return result;
}

//==============================================================================
// area
//==============================================================================
// In general, the area of a polygon in 3D space is non-trivial to compute.
// The exception is when the polygon is planar (or a triangle).
// Note: convex quadrilaterals and triangles use a more efficient formula.

template <Int N>
PURE HOSTDEV constexpr auto
area(PlanarLinearPolygon<N> const & p) noexcept -> Float
{
  // Shoelace forumla A = 1/2 * sum_{i=0}^{n-1} cross(p_i, p_{i+1})
  // p_n = p_0
  Float sum = (p[N - 1]).cross(p[0]); // cross(p_{n-1}, p_0), the last term
  for (Int i = 0; i < N - 1; ++i) {
    sum += (p[i]).cross(p[i + 1]);
  }
  ASSERT(sum > 0);
  return sum / 2;
}

template <Int N>
PURE HOSTDEV constexpr auto
area(PlanarQuadraticPolygon<N> const & q) noexcept -> Float
{
  // Geometric decomposition:
  // Area of the linear polygon plus the area enclosed by the quadratic edges.
  // Note, the enclosed area is not necessarily positive.
  Float result = area(linearPolygon(q));
  Int constexpr num_edges = polygonNumEdges<2, N>(); 
  for (Int i = 0; i < num_edges; ++i) {
    result += enclosedArea(q.getEdge(i));
  }
  ASSERT(result > 0);
  return result;
}

//==============================================================================
// centroid
//==============================================================================

template <Int N>
PURE HOSTDEV constexpr auto
centroid(PlanarLinearPolygon<N> const & p) noexcept -> Point2
{
  // Similar to the shoelace formula.
  // C = 1/6A * sum_{i=0}^{n-1} cross(p_i, p_{i+1}) * (p_i + p_{i+1})
  Float area_sum = (p[N - 1]).cross(p[0]); // p_{n-1} x p_0, the last term
  Point2 centroid_sum = area_sum * (p[N - 1] + p[0]);
  for (Int i = 0; i < N - 1; ++i) {
    Float const a = (p[i]).cross(p[i + 1]);
    area_sum += a;
    centroid_sum += a * (p[i] + p[i + 1]);
  }
  return centroid_sum / (static_cast<Float>(3) * area_sum);
}

template <Int N>
PURE HOSTDEV constexpr auto
centroid(PlanarQuadraticPolygon<N> const & q) noexcept -> Point2
{
  auto lin_poly = linearPolygon(q);
  Float area_sum = lin_poly.area();
  Point2 centroid_sum = area_sum * lin_poly.centroid();
  Int constexpr num_edges = polygonNumEdges<2, N>(); 
  for (Int i = 0; i < num_edges; ++i) {
    auto const e = q.getEdge(i);
    Float const a = enclosedArea(e);
    area_sum += a;
    centroid_sum += a * enclosedCentroid(e);
  }
  return centroid_sum / area_sum;
}

//==============================================================================
// contains
//==============================================================================

template <Int N>
PURE HOSTDEV constexpr auto
contains(PlanarQuadraticPolygon<N> const & poly, Point2 const p) noexcept -> bool
{
  Int constexpr m = polygonNumEdges<2, N>();
  // The point is inside the polygon if it is left of all the edges.
  for (Int i = 0; i < m; ++i) {
    if (!poly.getEdge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

//==============================================================================
// meanChordLength
//==============================================================================
// For a convex planar polygon, the mean chord length is simply pi * area / perimeter.
// De Kruijf, W. J. M., and J. L. Kloosterman.
// "On the average chord length in reactor physics." Annals of Nuclear Energy 30.5 (2003):
// 549-553.
//
// It can be shown that this is also true for a concave polygon.

template <Int P, Int N>
PURE HOSTDEV constexpr auto
meanChordLength(PlanarPolygon<P, N> const & p) noexcept -> Float
{
  auto const result = um2::pi<Float> * p.area() / p.perimeter();
  ASSERT(result > 0);
  return result;
}

//==============================================================================
// intersect
//==============================================================================

template <Int P, Int N>
HOSTDEV constexpr auto
intersect(PlanarPolygon<P, N> const & poly, Ray2 const & ray, Float * const buffer) noexcept -> Int
{
  Int constexpr m = polygonNumEdges<P, N>();
  Int hits = 0;
  for (Int i = 0; i < m; ++i) {
    hits += poly.getEdge(i).intersect(ray, buffer + hits);
  }
  return hits;
}

//==============================================================================
// hasSelfIntersection 
//==============================================================================
// Quadratic polygons only. 

template <Int N>
HOSTDEV constexpr auto
hasSelfIntersection(PlanarQuadraticPolygon<N> const & poly, Point2 * buffer) noexcept -> bool
{
  // Edge i should intersect edge i + 1 exactly once.
  Int constexpr m = polygonNumEdges<2, N>();
  for (Int i = 0; i < m - 1; ++i) {
    if (poly.getEdge(i).intersect(poly.getEdge(i + 1), buffer) > 1) {
      return true;
    }
  }
  
  // Edge m - 1 should intersect edge 0 exactly once.
  if (poly.getEdge(m - 1).intersect(poly.getEdge(0), buffer) > 1) {
    return true;
  }

  // If this is a quadratic quadrilateral, we need to check i vs i + 2.
  if constexpr (m == 4) {
    if (poly.getEdge(0).intersect(poly.getEdge(2), buffer) > 1) {
      return true;
    }
    if (poly.getEdge(1).intersect(poly.getEdge(3), buffer) > 1) {
      return true;
    }
  }

  return false;
}

template <Int N>
HOSTDEV constexpr auto
hasSelfIntersection(PlanarQuadraticPolygon<N> const & poly) noexcept -> bool
{
  Point2 buffer[2 * N];
  return hasSelfIntersection(poly, buffer);
}

} // namespace um2
