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

template <I K, I P, I N, I D>
class Polytope
{
  static_assert(K > 0 && K <= 3, "Polytope dimension must be 1, 2, or 3");
  Point<D> _v[N];
};

//==============================================================================
// Aliases
//==============================================================================

template <I K, I N, I D>
using LinearPolytope = Polytope<K, 1, N, D>;

// K-Polytopes
//-----------------------------------------------------------------------------
template <I P, I N, I D>
using Dion = Polytope<1, P, N, D>;
template <I P, I N, I D>
using Polygon = Polytope<2, P, N, D>;
template <I P, I N, I D>
using Polyhedron = Polytope<3, P, N, D>;

// Dions
//-----------------------------------------------------------------------------
template <I D>
using LineSegment = Dion<1, 2, D>;
template <I D>
using QuadraticSegment = Dion<2, 3, D>;

// Planar dions
template <I P, I N>
using PlanarDion = Dion<P, N, 2>;
template <I N>
using PlanarLineSegment = LineSegment<2>;
template <I N>
using PlanarQuadraticSegment = QuadraticSegment<2>;

// Dimension specific aliases
using LineSegment2 = LineSegment<2>;
using QuadraticSegment2 = QuadraticSegment<2>;

// Polygons
//-----------------------------------------------------------------------------
template <I N, I D>
using LinearPolygon = Polygon<1, N, D>;
template <I N, I D>
using QuadraticPolygon = Polygon<2, N, D>;

// Planar polygons
template <I P, I N>
using PlanarPolygon = Polygon<P, N, 2>;
template <I N>
using PlanarLinearPolygon = LinearPolygon<N, 2>;
template <I N>
using PlanarQuadraticPolygon = QuadraticPolygon<N, 2>;

// N-vertex polygons
template <I D>
using Triangle = LinearPolygon<3, D>;
template <I D>
using Quadrilateral = LinearPolygon<4, D>;
template <I D>
using QuadraticTriangle = QuadraticPolygon<6, D>;
template <I D>
using QuadraticQuadrilateral = QuadraticPolygon<8, D>;

// N-vertex polygons (shorthand)
template <I D>
using Tri = Triangle<D>;
template <I D>
using Quad = Quadrilateral<D>;
template <I D>
using Tri6 = QuadraticTriangle<D>;
template <I D>
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
template <I N, I D>
using LinearPolyhedron = Polyhedron<1, N, D>;
template <I N, I D>
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
template <I K, I N, I D>
PURE HOSTDEV constexpr auto
boundingBox(LinearPolytope<K, N, D> const & polytope) noexcept -> AxisAlignedBox<D>
{
  return boundingBox(polytope.vertices(), N);
}

} // namespace um2
