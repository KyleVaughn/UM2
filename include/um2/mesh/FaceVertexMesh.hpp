#pragma once

#include <um2/config.hpp>

#include <um2/common/Vector.hpp>

#include <um2/geometry/Point.hpp>

namespace um2
{

// FACE-VERTEX MESH
//-----------------------------------------------------------------------------
// A 2D volumetric or 3D surface mesh composed of polygones of polynomial order P.
// Each polygon (face) is composed of N vertices. Each vertex is a D-dimensional
// point of floating point type T. For heterogeneous meshes, N is the sum of the
// number of vertices of each face. This makes each mesh type uniquely
// identifiable by P and N as follows:
//  - P = 1, N =  3: Triangular mesh
//  - P = 1, N =  4: Quadrilateral mesh
//  - P = 1, N =  7: Tri + quad mesh
//  - P = 2, N =  6: Quadratic triangular mesh
//  - P = 2, N =  8: Quadratic quadrilateral mesh
//  - P = 2, N = 14: Quadratic Tri + quad mesh
// Let I be the signed integer type used to index vertices and faces.
// We will use some simple meshes to explain the data structure. A more detailed
// explanation of each member follows.
//  - A TriMesh (FaceVertexMesh<1, 3>) with two triangles:
//      3---2
//      | / |
//      0---1
//      vertices = { {0, 0}, {1, 0}, {1, 1}, {0, 1} }
//          4 vertices on the unit square
//      fv = { {0, 1, 2}, {2, 3, 0} }
//          The 6 vertex indices composing the two triangles {0, 1, 2} and {2, 3, 0}
//      vf = { 0, 1, 0, 0, 1, 1 }
//          The face indices to which each vertex belongs. More precisely, vertex
//          0 belongs to faces 0 and 1, vertex 1 belongs to face 0 only, etc.
//          Face IDs are ordered least to greatest.
//      vf_offsets = { 0, 2, 3, 5, 6}
//          vf_offsets[i] is the index of the smallest face ID to which vertex i
//          belongs. There is an additional element at the end, which is the length
//          of the vf vector. Used to calculate the number of faces to which each
//          vertex belongs.
//  - A TriQuadMesh (FaceVertexMesh<1, 7>) with one triangle and one quad:
//    3---2
//    |   | \                                                                       .
//    0---1---4
//      vertices = { {0, 0}, {1, 0}, {1, 1}, {0, 1}, {2, 0} }
//          5 vertices on the unit square and one on the unit line
//      fv = {{0, 1, 2, 3}, {-1, 1, 4, 2}
//          The 7 vertex indices composing the quad {0, 1, 2, 3} and the triangle
//          {1, 4, 2}, represented using a union of Vec<3> and Vec<4>. The first element
//          being negative tells us that the face is a triangle.
//      vf = { 0, 0, 1, 0, 1, 0, 1 }
//          v0 belongs to face 0, v1 belongs to faces 0 and 1, v2 belongs to faces
//          0 and 1, v3 belongs to face 0, v4 belongs to face 1.
//      vf_offsets = { 0, 1, 3, 5, 6, 7 }
//          v0's connectivity starts at vf[0] and has 1 - 0 = 1 face. v1's
//          connectivity starts at vf[1] and has 3 - 1 = 2 faces. etc.

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
struct FaceVertexMesh {
};

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <Size N, Size D, std::floating_point T, std::signed_integral I>
using LinearPolygonMesh = FaceVertexMesh<1, N, D, T, I>;

template <Size N, Size D, std::floating_point T, std::signed_integral I>
using QuadraticPolygonMesh = FaceVertexMesh<2, N, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using TriMesh = LinearPolygonMesh<3, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadMesh = LinearPolygonMesh<4, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using TriQuadMesh = LinearPolygonMesh<7, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticTriMesh = QuadraticPolygonMesh<6, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticQuadMesh = QuadraticPolygonMesh<8, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticTriQuadMesh = QuadraticPolygonMesh<14, D, T, I>;

} // namespace um2
