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

#define STATIC_ASSERT_VALID_POLYGON \
  static_assert(D >= 2, "Polygons must be embedded in at least 2 dimensions"); \
  static_assert(N >= 2, "A polygon must have at least 2 vertices");

namespace um2
{

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

//==============================================================================
// perimeter
//==============================================================================
// The perimeter of a polygon is the sum of the lengths of its edges.
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
  return result;
}

//==============================================================================
// boundingBox
//==============================================================================

//==============================================================================
// area
//==============================================================================
// In general, the area of a polygon in 3D space is non-trivial to compute.
// The exception is when the polygon is planar (or a triangle).
// Note: convex quadrilaterals and triangles use a more efficient formula.

//template <typename T>
//PURE HOSTDEV constexpr auto
//area(Quadrilateral2<T> const & q) noexcept -> T
//{
//  assert(isConvex(q));
//  // (v2 - v0).cross(v3 - v1) / 2
//  Vec2<T> const v20 = q[2] - q[0];
//  Vec2<T> const v31 = q[3] - q[1];
//  return v20.cross(v31) / 2;
//}
//
// Area of a planar linear polygon
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
  return sum / 2;
}

//template <Size N, typename T>
//PURE HOSTDEV constexpr auto
//area(PlanarQuadraticPolygon<N, T> const & q) noexcept -> T
//{
//  T result = area(linearPolygon(q));
//  Size constexpr num_edges = PlanarQuadraticPolygon<N, T>::numEdges();
//  for (Size i = 0; i < num_edges; ++i) {
//    result += enclosedArea(getEdge(q, i));
//  }
//  return result;
//}
//

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

//template <Size N, typename T>
//PURE HOSTDEV constexpr auto
//centroid(PlanarQuadraticPolygon<N, T> const & q) noexcept -> Point2<T>
//{
//  auto lin_poly = linearPolygon(q);
//  T area_sum = lin_poly.area();
//  Point2<T> centroid_sum = area_sum * centroid(lin_poly);
//  Size constexpr num_edges = PlanarQuadraticPolygon<N, T>::numEdges();
//  for (Size i = 0; i < num_edges; ++i) {
//    auto const e = getEdge(q, i);
//    T const a = enclosedArea(e);
//    area_sum += a;
//    centroid_sum += a * enclosedCentroid(e);
//  }
//  return centroid_sum / area_sum;
//}

//==============================================================================
// contains 
//==============================================================================

//template <Int N>
//PURE HOSTDEV constexpr auto
//contains(PlanarLinearPolygon<N> const & poly, Point2 const p) noexcept -> bool
//{
//
//}

//==============================================================================    
// meanChordLength    
//==============================================================================    
// For a convex planar polygon, the mean chord length is simply pi * area / perimeter.    
// De Kruijf, W. J. M., and J. L. Kloosterman.    
// "On the average chord length in reactor physics." Annals of Nuclear Energy 30.5 (2003):    
// 549-553.    
    
template <Int P, Int N>    
PURE HOSTDEV constexpr auto    
meanChordLength(PlanarPolygon<P, N> const & p) noexcept -> Float
{    
  return um2::pi<Float> * p.area() / p.perimeter();    
} 

//==============================================================================
// intersect
//==============================================================================

template <Int P, Int N>
HOSTDEV constexpr auto
intersect(PlanarPolygon<P, N> const & poly, Ray2 const & ray, Float * buffer) noexcept -> Int
{
  Int hits = 0;
  for (Int i = 0; i < N; ++i) {
    hits += poly.getEdge(i).intersect(ray, buffer + hits);
  }
  return hits;
}

} // namespace um2
