#pragma once

#include <um2/geometry/Polygon.hpp>
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/mesh/MeshFile.hpp>
#include <um2/stdlib/Vector.hpp>

namespace um2
{

//=============================================================================
// FACE-VERTEX MESH
//=============================================================================
//
// A 2D volumetric or 3D surface mesh composed of polygons of polynomial order P.
// Each polygon (face) is composed of N vertices. Each vertex is a D-dimensional
// point of floating point type T.
//  - P = 1, N = 3: Triangular mesh
//  - P = 1, N = 4: Quadrilateral mesh
//  - P = 2, N = 6: Quadratic triangular mesh
//  - P = 2, N = 8: Quadratic quadrilateral mesh
//
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
//
template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
struct FaceVertexMesh {

  using FaceConn = Vec<N, I>;
  using Face = Polygon<P, N, D, T>;

  Vector<Point<D, T>> vertices;
  Vector<FaceConn> fv;
  Vector<I> vf_offsets; // size = num_vertices + 1
  Vector<I> vf;         // size = vf_offsets[num_vertices]

  //===========================================================================
  // Constructors
  //===========================================================================

  constexpr FaceVertexMesh() noexcept = default;

  explicit FaceVertexMesh(MeshFile<T, I> const & file);

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numVertices() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numFaces() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFace(Size i) const noexcept -> Face;

  //===========================================================================
  // Methods
  //===========================================================================

  PURE [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE [[nodiscard]] constexpr auto
  faceContaining(Point<D, T> const & p) const noexcept -> Size;

  void
  flipFace(Size i) noexcept;

  void
  toMeshFile(MeshFile<T, I> & file) const noexcept;

  [[nodiscard]] constexpr auto
  getFaceAreas() const noexcept -> Vector<T>;

  void
  intersect(Ray<D, T> const & ray, T * intersections, Size * n) const noexcept;
};

//==============================================================================
// Aliases
//==============================================================================

// Polynomial order
template <Size N, Size D, std::floating_point T, std::signed_integral I>
using LinearPolygonMesh = FaceVertexMesh<1, N, D, T, I>;
template <Size N, Size D, std::floating_point T, std::signed_integral I>
using QuadraticPolygonMesh = FaceVertexMesh<2, N, D, T, I>;

// Number of vertices per face
template <Size D, std::floating_point T, std::signed_integral I>
using TriMesh = LinearPolygonMesh<3, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using QuadMesh = LinearPolygonMesh<4, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticTriMesh = QuadraticPolygonMesh<6, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticQuadMesh = QuadraticPolygonMesh<8, D, T, I>;

// 2D
template <Size P, Size N, std::floating_point T, std::signed_integral I>
using PlanarPolygonMesh = FaceVertexMesh<P, N, 2, T, I>;
template <Size N, std::floating_point T, std::signed_integral I>
using PlanarLinearPolygonMesh = FaceVertexMesh<1, N, 2, T, I>;
template <Size N, std::floating_point T, std::signed_integral I>
using PlanarQuadraticPolygonMesh = FaceVertexMesh<2, N, 2, T, I>;

} // namespace um2

#include "FaceVertexMesh.inl"
