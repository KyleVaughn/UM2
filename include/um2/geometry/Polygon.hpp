#pragma once

#include <um2/geometry/Dion.hpp>

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

namespace um2
{

template <Size P, Size N, Size D, typename T>
struct Polytope<2, P, N, D, T> {

  Point<D, T> v[N];

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Point<D, T> &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Point<D, T> const &;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
    requires(sizeof...(Pts) == N  && (std::same_as<Point<D, T>, Pts> && ...))
  // NOLINTNEXTLINE(google-explicit-constructor) justification: implicit conversion
  HOSTDEV constexpr Polytope(Pts const... args) noexcept : v{args...} {}

  //==============================================================================
  // Methods
  //==============================================================================

  template <typename R, typename S>
  PURE HOSTDEV constexpr auto
  operator()(R r, S s) const noexcept -> Point<D, T>;

  template <typename R, typename S>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r, S s) const noexcept -> Mat<D, 2, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto    
  getEdge(Size i) const noexcept -> LineSegment<D, T>
  requires(P == 1);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Size i) const noexcept -> QuadraticSegment<D, T>
  requires(P == 2);
    
  PURE HOSTDEV [[nodiscard]] constexpr auto    
  contains(Point<D, T> const & p) const noexcept -> bool;    

  PURE HOSTDEV [[nodiscard]] constexpr auto    
  area() const noexcept -> T;    
    
  PURE HOSTDEV [[nodiscard]] constexpr auto    
  centroid() const noexcept -> Point<D, T>;    
    
  PURE HOSTDEV [[nodiscard]] constexpr auto    
  boundingBox() const noexcept -> AxisAlignedBox<D, T>; 

  PURE HOSTDEV [[nodiscard]] constexpr auto    
  isCCW() const noexcept -> bool;

}; // Polygon

//==============================================================================
// polygonNumEdges
//==============================================================================

template <Size P, Size N>
PURE HOSTDEV constexpr auto
polygonNumEdges() noexcept -> Size
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
  Size const num_edges = polygonNumEdges<2, N>();
  for (Size i = 1; i < num_edges; ++i) {
    box += boundingBox(getEdge(p, i));
  }
  return box;
}

} // namespace um2

#include "Polygon.inl"
