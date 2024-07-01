#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>

//==============================================================================
// POLYTOPE
//==============================================================================
// A K-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points.
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

template <Int K, Int P, Int N, Int D, class T>
class Polytope
{
  static_assert(K > 0 && K <= 3, "Polytope dimension must be 1, 2, or 3");
};

//==============================================================================
// Aliases/Specializations
//==============================================================================

template <Int K, Int N, Int D, class T>
using LinearPolytope = Polytope<K, 1, N, D, T>;

// K-Polytopes
//-----------------------------------------------------------------------------
template <Int P, Int N, Int D, class T>
using Dion = Polytope<1, P, N, D, T>;
template <Int P, Int N, Int D, class T>
using Polygon = Polytope<2, P, N, D, T>;
template <Int P, Int N, Int D, class T>
using Polyhedron = Polytope<3, P, N, D, T>;

// Dions
//-----------------------------------------------------------------------------
template <Int D, class T>
using LineSegment = Dion<1, 2, D, T>;
template <Int D, class T>
using QuadraticSegment = Dion<2, 3, D, T>;

// Planar dions
template <Int P, Int N, class T>
using PlanarDion = Dion<P, N, 2, T>;
template <Int N, class T>
using PlanarLineSegment = LineSegment<2, T>;
template <Int N, class T>
using PlanarQuadraticSegment = QuadraticSegment<2, T>;

// Dimension specific aliases
template <class T>
using LineSegment2 = LineSegment<2, T>;

template <class T>
using QuadraticSegment2 = QuadraticSegment<2, T>;

// Polygons
//-----------------------------------------------------------------------------
template <Int N, Int D, class T>
using LinearPolygon = Polygon<1, N, D, T>;
template <Int N, Int D, class T>
using QuadraticPolygon = Polygon<2, N, D, T>;

// Planar polygons
template <Int P, Int N, class T>
using PlanarPolygon = Polygon<P, N, 2, T>;
template <Int N, class T>
using PlanarLinearPolygon = LinearPolygon<N, 2, T>;
template <Int N, class T>
using PlanarQuadraticPolygon = QuadraticPolygon<N, 2, T>;

// N-vertex polygons
template <Int D, class T>
using Triangle = LinearPolygon<3, D, T>;
template <Int D, class T>
using Quadrilateral = LinearPolygon<4, D, T>;
template <Int D, class T>
using QuadraticTriangle = QuadraticPolygon<6, D, T>;
template <Int D, class T>
using QuadraticQuadrilateral = QuadraticPolygon<8, D, T>;

// N-vertex polygons (shorthand)
template <Int D, class T>
using Tri = Triangle<D, T>;
template <Int D, class T>
using Quad = Quadrilateral<D, T>;
template <Int D, class T>
using Tri6 = QuadraticTriangle<D, T>;
template <Int D, class T>
using Quad8 = QuadraticQuadrilateral<D, T>;

// Dimension specific aliases
template <class T>
using Triangle2 = Triangle<2, T>;
template <class T>
using Triangle3 = Triangle<3, T>;
template <class T>
using Quadrilateral2 = Quadrilateral<2, T>;
template <class T>
using Quadrilateral3 = Quadrilateral<3, T>;
template <class T>
using QuadraticTriangle2 = QuadraticTriangle<2, T>;
template <class T>
using QuadraticTriangle3 = QuadraticTriangle<3, T>;
template <class T>
using QuadraticQuadrilateral2 = QuadraticQuadrilateral<2, T>;
template <class T>
using QuadraticQuadrilateral3 = QuadraticQuadrilateral<3, T>;

using Triangle2F = Triangle2<Float>;
using Triangle3F = Triangle3<Float>;
using Quadrilateral2F = Quadrilateral2<Float>;
using Quadrilateral3F = Quadrilateral3<Float>;
using QuadraticTriangle2F = QuadraticTriangle2<Float>;
using QuadraticTriangle3F = QuadraticTriangle3<Float>;
using QuadraticQuadrilateral2F = QuadraticQuadrilateral2<Float>;
using QuadraticQuadrilateral3F = QuadraticQuadrilateral3<Float>;

// Polyhedrons
//-----------------------------------------------------------------------------
template <Int N, Int D, class T>
using LinearPolyhedron = Polyhedron<1, N, D, T>;

template <Int N, Int D, class T>
using QuadraticPolyhedron = Polyhedron<2, N, D, T>;

// N-vertex polyhedrons
// Only allow embedding in 3D for now
template <class T>
using Tetrahedron = LinearPolyhedron<4, 3, T>;
template <class T>
using Hexahedron = LinearPolyhedron<8, 3, T>;
template <class T>
using QuadraticTetrahedron = QuadraticPolyhedron<10, 3, T>;
template <class T>
using QuadraticHexahedron = QuadraticPolyhedron<20, 3, T>;

//==============================================================================
// Functions
//==============================================================================

// The bounding box of any linear polytope is the bounding box of its vertices.
template <Int K, Int N, Int D, class T>
PURE HOSTDEV constexpr auto
boundingBox(LinearPolytope<K, N, D, T> const & polytope) noexcept -> AxisAlignedBox<D, T>
{
  return boundingBox(polytope.vertices(), N);
}

} // namespace um2
