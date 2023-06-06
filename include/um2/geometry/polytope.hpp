#pragma once

#include <um2/common/config.hpp>

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
template <len_t K, len_t P, len_t N, len_t D, typename T>
// NOLINTNEXTLINE(bugprone-reserved-identifier, readability-identifier-naming)
struct Polytope;

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <len_t P, len_t N, len_t D, typename T>
using Dion = Polytope<1, P, N, D, T>;

template <len_t P, len_t N, len_t D, typename T>
using Polygon = Polytope<2, P, N, D, T>;

template <len_t P, len_t N, len_t D, typename T>
using Polyhedron = Polytope<3, P, N, D, T>;

// Dions
template <len_t D, typename T>
using LineSegment = Dion<1, 2, D, T>;

template <len_t D, typename T>
using QuadraticSegment = Dion<2, 3, D, T>;

// Polygons
template <len_t N, len_t D, typename T>
using LinearPolygon = Polygon<1, N, D, T>;

template <len_t N, len_t D, typename T>
using QuadraticPolygon = Polygon<2, N, D, T>;

template <len_t D, typename T>
using Triangle = LinearPolygon<3, D, T>;

template <len_t D, typename T>
using Quadrilateral = LinearPolygon<4, D, T>;

template <len_t D, typename T>
using QuadraticTriangle = QuadraticPolygon<6, D, T>;

template <len_t D, typename T>
using QuadraticQuadrilateral = QuadraticPolygon<8, D, T>;

// Polyhedrons
template <len_t N, len_t D, typename T>
using LinearPolyhedron = Polyhedron<1, N, D, T>;

template <len_t N, len_t D, typename T>
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

} // namespace um2
