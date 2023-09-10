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

  using Edge = Dion<P, P + 1, D, T>;

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
    requires(sizeof...(Pts) == N && (std::same_as<Point<D, T>, Pts> && ...))
  // NOLINTBEGIN(google-explicit-constructor) justification: implicit conversion
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : v{args...}
  {
  }
  // NOLINTEND(google-explicit-constructor)

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
  getEdge(Size i) const noexcept -> Edge;

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
// Methods
//==============================================================================

template <Size P, Size N>
PURE HOSTDEV constexpr auto
polygonNumEdges() noexcept -> Size
{
  static_assert(P == 1 || P == 2, "Only P = 1 or P = 2 supported");
  return N / P;
}

//==============================================================================
// interpolate
//==============================================================================

template <Size P, Size N, Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Polygon<P, N, D, T> const & poly, R r, S s) noexcept -> Point<D, T>;

//==============================================================================
// jacobian
//==============================================================================

template <Size P, Size N, Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Polygon<P, N, D, T> const & poly, R r, S s) noexcept -> Mat<D, 2, T>;

//==============================================================================
// getEdge
//==============================================================================

template <Size N, Size D, typename T>    
PURE HOSTDEV constexpr auto    
getEdge(LinearPolygon<N, D, T> const & p, Size i) noexcept -> LineSegment<D, T>;

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
getEdge(QuadraticPolygon<N, D, T> const & p, Size i) noexcept -> QuadraticSegment<D, T>;

//==============================================================================
// contains
//==============================================================================

template <Size N, typename T>    
PURE HOSTDEV constexpr auto    
contains(PlanarLinearPolygon<N, T> const & poly, Point2<T> const & p) noexcept -> bool;

template <Size N, typename T>
PURE HOSTDEV constexpr auto
contains(PlanarQuadraticPolygon<N, T> const & q, Point2<T> const & p) noexcept -> bool;

//==============================================================================
// area
//==============================================================================

template <typename T>
PURE HOSTDEV constexpr auto
area(Triangle3<T> const & tri) noexcept -> T;


template <Size N, typename T>    
PURE HOSTDEV constexpr auto    
area(PlanarLinearPolygon<N, T> const & p) noexcept -> T;

template <Size N, typename T>    
PURE HOSTDEV constexpr auto    
area(PlanarQuadraticPolygon<N, T> const & q) noexcept -> T;

} // namespace um2

#include "Polygon.inl"
