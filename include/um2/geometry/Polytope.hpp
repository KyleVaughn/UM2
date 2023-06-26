#pragma once

#include <um2/config.hpp>
#include <um2/geometry/AxisAlignedBox.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// POLYTOPE
// -----------------------------------------------------------------------------
//
// A K-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points of type T.
//
// This struct only supports the shapes found in "The Visualization Toolkit:
// An Object-Oriented Approach to 3D Graphics, 4th Edition, Chapter 8, Advanced
// Data Representation".
//
// See the VTK book for specific vertex ordering info, but generally vertices are
// ordered in a counterclockwise fashion, with vertices of the linear shape given
// first.
//
// See https://en.wikipedia.org/wiki/Polytope for help with terminology.
//
template <Size K, Size P, Size N, Size D, typename T>
struct Polytope;

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <Size P, Size N, Size D, typename T>
using Dion = Polytope<1, P, N, D, T>;

template <Size P, Size N, Size D, typename T>
using Polygon = Polytope<2, P, N, D, T>;

template <Size P, Size N, Size D, typename T>
using Polyhedron = Polytope<3, P, N, D, T>;

// Dions
template <Size D, typename T>
using LineSegment = Dion<1, 2, D, T>;

template <Size D, typename T>
using QuadraticSegment = Dion<2, 3, D, T>;

// Polygons
template <Size N, Size D, typename T>
using LinearPolygon = Polygon<1, N, D, T>;

template <Size N, Size D, typename T>
using QuadraticPolygon = Polygon<2, N, D, T>;

template <Size D, typename T>
using Triangle = LinearPolygon<3, D, T>;

template <Size D, typename T>
using Quadrilateral = LinearPolygon<4, D, T>;

template <Size D, typename T>
using QuadraticTriangle = QuadraticPolygon<6, D, T>;

template <Size D, typename T>
using QuadraticQuadrilateral = QuadraticPolygon<8, D, T>;

// Polyhedrons
template <Size N, Size D, typename T>
using LinearPolyhedron = Polyhedron<1, N, D, T>;

template <Size N, Size D, typename T>
using QuadraticPolyhedron = Polyhedron<2, N, D, T>;

// Only allow embedding in 3D for now
template <typename T>
using Tetrahedron = LinearPolyhedron<4, 3, T>;

template <typename T>
using Hexahedron = LinearPolyhedron<8, 3, T>;

template <typename T>
using QuadraticTetrahedron = QuadraticPolyhedron<10, 3, T>;

template <typename T>
using QuadraticHexahedron = QuadraticPolyhedron<20, 3, T>;

// -----------------------------------------------------------------------------
// General functions
// -----------------------------------------------------------------------------
// Some functions may be easily generalized to subsets of polytopes, hence we define
// them once here instead of repeating them for each specific polytope.

// -----------------------------------------------------------------------------
// area
// -----------------------------------------------------------------------------
template <typename T>
PURE HOSTDEV constexpr auto
triangleArea(Point2<T> const & p0, Point2<T> const & p1, Point2<T> const & p2) noexcept
    -> T;

template <typename T>
PURE HOSTDEV constexpr auto
quadrilateralArea(Point2<T> const & p0, Point2<T> const & p1, Point2<T> const & p2,
                  Point2<T> const & p3) noexcept -> T;

// Area of a linear, planar polygon
template <Size N, typename T>
PURE HOSTDEV constexpr auto
area(LinearPolygon<N, 2, T> const & poly) noexcept -> T;

template <typename T>
PURE HOSTDEV constexpr auto
area(Triangle<3, T> const & tri) noexcept -> T;

// -----------------------------------------------------------------------------
// centroid
// -----------------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
triangleCentroid(Point<D, T> const & p0, Point<D, T> const & p1,
                 Point<D, T> const & p2) noexcept -> Point<D, T>;

template <typename T>
PURE HOSTDEV constexpr auto
quadrilateralCentroid(Point2<T> const & p0, Point2<T> const & p1, Point2<T> const & p2,
                      Point2<T> const & p3) noexcept -> Point2<T>;

// Centroid of a linear, planar polygon
template <Size N, typename T>
PURE HOSTDEV constexpr auto
centroid(LinearPolygon<N, 2, T> const & poly) noexcept -> Point2<T>;

// -----------------------------------------------------------------------------
// boundingBox
// -----------------------------------------------------------------------------
// The bounding box of any linear polytope is simply the min and max of its
// vertices.
template <Size K, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(Polytope<K, 1, N, D, T> const & poly) -> AxisAlignedBox<D, T>;

} // namespace um2

#include "Polytope.inl"
