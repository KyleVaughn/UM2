#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>

namespace um2
{

//==============================================================================
// POLYTOPE
//==============================================================================
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

template <Size K, Size P, Size N, Size D, typename T>
struct Polytope {
  static_assert(K > 0 && K <= 3, "Polytope dimension must be 1, 2, or 3");
  Point<D, T> v[N];
};

//==============================================================================
// Aliases
//==============================================================================

template <Size K, Size N, Size D, typename T>
using LinearPolytope = Polytope<K, 1, N, D, T>;

// K-Polytopes
//-----------------------------------------------------------------------------
template <Size P, Size N, Size D, typename T>
using Dion = Polytope<1, P, N, D, T>;
template <Size P, Size N, Size D, typename T>
using Polygon = Polytope<2, P, N, D, T>;
template <Size P, Size N, Size D, typename T>
using Polyhedron = Polytope<3, P, N, D, T>;

// Dions
//-----------------------------------------------------------------------------
template <Size D, typename T>
using LineSegment = Dion<1, 2, D, T>;
template <Size D, typename T>
using QuadraticSegment = Dion<2, 3, D, T>;

// Planar dions
template <Size P, Size N, typename T>
using PlanarDion = Dion<P, N, 2, T>;
template <Size N, typename T>
using PlanarLineSegment = LineSegment<2, T>;
template <Size N, typename T>
using PlanarQuadraticSegment = QuadraticSegment<2, T>;

// Dimension/data type specific aliases
template <typename T>
using LineSegment2 = LineSegment<2, T>;
using LineSegment2f = LineSegment2<float>;
using LineSegment2d = LineSegment2<double>;
template <typename T>
using QuadraticSegment2 = QuadraticSegment<2, T>;
using QuadraticSegment2f = QuadraticSegment2<float>;
using QuadraticSegment2d = QuadraticSegment2<double>;

// Polygons
//-----------------------------------------------------------------------------
template <Size N, Size D, typename T>
using LinearPolygon = Polygon<1, N, D, T>;
template <Size N, Size D, typename T>
using QuadraticPolygon = Polygon<2, N, D, T>;

// Planar polygons
template <Size P, Size N, typename T>
using PlanarPolygon = Polygon<P, N, 2, T>;
template <Size N, typename T>
using PlanarLinearPolygon = LinearPolygon<N, 2, T>;
template <Size N, typename T>
using PlanarQuadraticPolygon = QuadraticPolygon<N, 2, T>;

// N-vertex polygons
template <Size D, typename T>
using Triangle = LinearPolygon<3, D, T>;
template <Size D, typename T>
using Quadrilateral = LinearPolygon<4, D, T>;
template <Size D, typename T>
using QuadraticTriangle = QuadraticPolygon<6, D, T>;
template <Size D, typename T>
using QuadraticQuadrilateral = QuadraticPolygon<8, D, T>;

// Dimension/data type specific aliases
template <typename T>
using Triangle2 = Triangle<2, T>;
using Triangle2f = Triangle2<float>;
using Triangle2d = Triangle2<double>;
template <typename T>
using Triangle3 = Triangle<3, T>;
using Triangle3f = Triangle3<float>;
using Triangle3d = Triangle3<double>;
template <typename T>
using Quadrilateral2 = Quadrilateral<2, T>;
using Quadrilateral2f = Quadrilateral2<float>;
using Quadrilateral2d = Quadrilateral2<double>;
template <typename T>
using QuadraticTriangle2 = QuadraticTriangle<2, T>;
using QuadraticTriangle2f = QuadraticTriangle2<float>;
using QuadraticTriangle2d = QuadraticTriangle2<double>;
template <typename T>
using QuadraticTriangle3 = QuadraticTriangle<3, T>;
using QuadraticTriangle3f = QuadraticTriangle3<float>;
using QuadraticTriangle3d = QuadraticTriangle3<double>;
template <typename T>
using QuadraticQuadrilateral2 = QuadraticQuadrilateral<2, T>;
using QuadraticQuadrilateral2f = QuadraticQuadrilateral2<float>;
using QuadraticQuadrilateral2d = QuadraticQuadrilateral2<double>;
template <typename T>
using QuadraticQuadrilateral3 = QuadraticQuadrilateral<3, T>;
using QuadraticQuadrilateral3f = QuadraticQuadrilateral3<float>;
using QuadraticQuadrilateral3d = QuadraticQuadrilateral3<double>;

// Polyhedrons
//-----------------------------------------------------------------------------
template <Size N, Size D, typename T>
using LinearPolyhedron = Polyhedron<1, N, D, T>;
template <Size N, Size D, typename T>
using QuadraticPolyhedron = Polyhedron<2, N, D, T>;

// N-vertex polyhedrons
// Only allow embedding in 3D for now
template <typename T>
using Tetrahedron = LinearPolyhedron<4, 3, T>;
template <typename T>
using Hexahedron = LinearPolyhedron<8, 3, T>;
template <typename T>
using QuadraticTetrahedron = QuadraticPolyhedron<10, 3, T>;
template <typename T>
using QuadraticHexahedron = QuadraticPolyhedron<20, 3, T>;

//==============================================================================
// Methods
//==============================================================================

template <Size K, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(LinearPolytope<K, N, D, T> const & polytope) noexcept -> AxisAlignedBox<D, T>
{
  return boundingBox(polytope.v);
}

} // namespace um2
