#pragma once

#include <um2/geometry/LineSegment.hpp>
#include <um2/geometry/QuadraticSegment.hpp>

//==============================================================================
// Polygon
//==============================================================================
//
// A 2-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points of type T.
//
// For Polygons
//   Triangle (P = 1, N = 3)
//   Quadrilateral (P = 1, N = 4)
//   Quadratic Triangle (P = 2, N = 6)
//   Quadratic Quadrilateral (P = 2, N = 8)
// Defines:
//   interpolate
//   jacobian
//   getEdge
//   numEdges
//   linearPolygon
//   isConvex (Quadrilateral only)
//   area
//   centroid
//   contains (point)
//   boundingBox
//   isCCW
//   flipFace

#include <um2/geometry/polygon/area.inl>
#include <um2/geometry/polygon/centroid.inl>
#include <um2/geometry/polygon/contains.inl>
#include <um2/geometry/polygon/interpolate.inl>
#include <um2/geometry/polygon/jacobian.inl>
namespace um2
{

//// of its vertices. These 3 vertices are D-dimensional points of type T.    
//    
//template <Size P, Size N, Size D, typename T>    
//struct Polytope<2, P, N, D, T> {    
//
//  Point<D, T> v[N];
//
//}


//==============================================================================
// getEdge
//==============================================================================

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
getEdge(LinearPolygon<N, D, T> const & p, Size const i) noexcept
{
  assert(0 <= i && i < N);
  return (i < N - 1) ? LineSegment<D, T>(p[i], p[i + 1])
                     : LineSegment<D, T>(p[N - 1], p[0]);
}

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
getEdge(QuadraticPolygon<N, D, T> const & p, Size const i) noexcept
{
  assert(0 <= i && i < N);
  constexpr Size m = N / 2;
  return (i < m - 1) ? QuadraticSegment<D, T>(p[i], p[i + 1], p[i + m])
                     : QuadraticSegment<D, T>(p[m - 1], p[0], p[N - 1]);
}

//==============================================================================
// numEdges
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
numEdges(Polygon<P, N, D, T> const & /*p*/) noexcept -> Size
{
  static_assert(P == 1 || P == 2, "Only P = 1 or P = 2 supported");
  return N / P;
}

//==============================================================================
// linearPolygon
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticTriangle<D, T> const & q) noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(q[0], q[1], q[2]);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticQuadrilateral<D, T> const & q) noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(q[0], q[1], q[2], q[3]);
}

//==============================================================================
// isConvex
//==============================================================================

template <typename T>
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

//==============================================================================
// boundingBox
//==============================================================================

template <Size N, typename T>
PURE HOSTDEV constexpr auto
boundingBox(PlanarQuadraticPolygon<N, T> const & p) noexcept -> AxisAlignedBox2<T>
{
  AxisAlignedBox2<T> box = boundingBox(getEdge(p, 0));
  for (Size i = 1; i < numEdges(p); ++i) {
    box += boundingBox(getEdge(p, i));
  }
  return box;
}

//==============================================================================
// isCCW
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(Triangle2<T> const & t) noexcept -> bool
{
  return areCCW(t[0], t[1], t[2]);
}

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(Quadrilateral2<T> const & q) noexcept -> bool
{
  bool const b0 = areCCW(q[0], q[1], q[2]);
  bool const b1 = areCCW(q[0], q[2], q[3]);
  return b0 && b1;
}

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(QuadraticTriangle2<T> const & q) noexcept -> bool
{
  return isCCW(linearPolygon(q));
}

template <typename T>
PURE HOSTDEV constexpr auto
isCCW(QuadraticQuadrilateral2<T> const & q) noexcept -> bool
{
  return isCCW(linearPolygon(q));
}

//==============================================================================
// flipFace
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(Triangle<D, T> & t) noexcept
{
  um2::swap(t[1], t[2]);
}

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(Quadrilateral<D, T> & q) noexcept
{
  um2::swap(q[1], q[3]);
}

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(QuadraticTriangle<D, T> & q) noexcept
{
  um2::swap(q[1], q[2]);
  um2::swap(q[3], q[5]);
}

template <Size D, typename T>
HOSTDEV constexpr void
flipFace(QuadraticQuadrilateral<D, T> & q) noexcept
{
  um2::swap(q[1], q[3]);
  um2::swap(q[4], q[7]);
}

} // namespace um2
