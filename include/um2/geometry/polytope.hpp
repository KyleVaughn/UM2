#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>

//==============================================================================
// POLYTOPE
//==============================================================================
// A K-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points of type F.
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

namespace um2
{

template <Size K, Size P, Size N, Size D>
class Polytope
{
  static_assert(K > 0 && K <= 3, "Polytope dimension must be 1, 2, or 3");
  Point<D> _v[N];
};

//==============================================================================
// Aliases
//==============================================================================

template <Size K, Size N, Size D>
using LinearPolytope = Polytope<K, 1, N, D>;

// K-Polytopes
//-----------------------------------------------------------------------------
template <Size P, Size N, Size D>
using Dion = Polytope<1, P, N, D>;
template <Size P, Size N, Size D>
using Polygon = Polytope<2, P, N, D>;
template <Size P, Size N, Size D>
using Polyhedron = Polytope<3, P, N, D>;

// Dions
//-----------------------------------------------------------------------------
template <Size D>
using LineSegment = Dion<1, 2, D>;
template <Size D>
using QuadraticSegment = Dion<2, 3, D>;

// Planar dions
template <Size P, Size N>
using PlanarDion = Dion<P, N, 2>;
template <Size N>
using PlanarLineSegment = LineSegment<2>;
template <Size N>
using PlanarQuadraticSegment = QuadraticSegment<2>;

// Dimension specific aliases
using LineSegment2 = LineSegment<2>;
using QuadraticSegment2 = QuadraticSegment<2>;

// Polygons
//-----------------------------------------------------------------------------
template <Size N, Size D>
using LinearPolygon = Polygon<1, N, D>;
template <Size N, Size D>
using QuadraticPolygon = Polygon<2, N, D>;

// Planar polygons
template <Size P, Size N>
using PlanarPolygon = Polygon<P, N, 2>;
template <Size N>
using PlanarLinearPolygon = LinearPolygon<N, 2>;
template <Size N>
using PlanarQuadraticPolygon = QuadraticPolygon<N, 2>;

// N-vertex polygons
template <Size D>
using Triangle = LinearPolygon<3, D>;
template <Size D>
using Quadrilateral = LinearPolygon<4, D>;
template <Size D>
using QuadraticTriangle = QuadraticPolygon<6, D>;
template <Size D>
using QuadraticQuadrilateral = QuadraticPolygon<8, D>;

// N-vertex polygons (shorthand)
template <Size D>
using Tri = Triangle<D>;
template <Size D>
using Quad = Quadrilateral<D>;
template <Size D>
using Tri6 = QuadraticTriangle<D>;
template <Size D>
using Quad8 = QuadraticQuadrilateral<D>;

// Dimension specific aliases
using Triangle2 = Triangle<2>;
using Triangle3 = Triangle<3>;
using Quadrilateral2 = Quadrilateral<2>;
using Quadrilateral3 = Quadrilateral<3>;
using QuadraticTriangle2 = QuadraticTriangle<2>;
using QuadraticTriangle3 = QuadraticTriangle<3>;
using QuadraticQuadrilateral2 = QuadraticQuadrilateral<2>;
using QuadraticQuadrilateral3 = QuadraticQuadrilateral<3>;

// Polyhedrons
//-----------------------------------------------------------------------------
template <Size N, Size D>
using LinearPolyhedron = Polyhedron<1, N, D>;
template <Size N, Size D>
using QuadraticPolyhedron = Polyhedron<2, N, D>;

// N-vertex polyhedrons
// Only allow embedding in 3D for now
using Tetrahedron = LinearPolyhedron<4, 3>;
using Hexahedron = LinearPolyhedron<8, 3>;
using QuadraticTetrahedron = QuadraticPolyhedron<10, 3>;
using QuadraticHexahedron = QuadraticPolyhedron<20, 3>;

//==============================================================================
// Methods
//==============================================================================

// The bounding box of any linear polytope is the bounding box of its vertices.
template <Size K, Size N, Size D>
PURE HOSTDEV constexpr auto
boundingBox(LinearPolytope<K, N, D> const & polytope) noexcept -> AxisAlignedBox<D>
{
  return boundingBox(polytope.vertices(), N);
}

} // namespace um2
