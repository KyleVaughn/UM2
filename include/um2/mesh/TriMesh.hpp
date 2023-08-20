#pragma once

#include <um2/geometry/Triangle.hpp>
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/mesh/FaceVertexMesh.hpp>
#include <um2/mesh/MeshFile.hpp>
#include <um2/mesh/RegularPartition.hpp>

//=============================================================================
// TRI MESH 
//=============================================================================
//
// A 2D volumetric or 3D surface mesh composed of triangles. 
// This is a specialization of FaceVertexMesh for P=1 and N=3.

namespace um2
{

template <Size D, std::floating_point T, std::signed_integral I>
struct FaceVertexMesh<1, 3, D, T, I> {

  using FaceConn = Vec<3, I>;
  using Face = Triangle<D, T>;

  Vector<Point<D, T>> vertices;
  Vector<FaceConn> fv;
  Vector<I> vf_offsets; // size = num_vertices + 1
  Vector<I> vf;         // size = vf_offsets[num_vertices]

  //==============================================================================
  // Constructors
  //==============================================================================

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

  //==============================================================================
  // Methods
  //==============================================================================

  PURE [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE [[nodiscard]] constexpr auto
  faceContaining(Point<D, T> const & p) const noexcept -> Size;

  void
  flipFace(Size i) noexcept;

  void
  toMeshFile(MeshFile<T, I> & file) const noexcept;
};

} // namespace um2

#include "TriMesh.inl"
