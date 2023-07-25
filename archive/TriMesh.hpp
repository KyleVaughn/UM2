#pragma once

#include <um2/geometry/Triangle.hpp>
#include <um2/mesh/FaceVertexMesh.hpp>
#include <um2/mesh/RegularPartition.hpp>

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

  // --------------------------------------------------------------------------
  // Constructors
  // --------------------------------------------------------------------------

  constexpr FaceVertexMesh() noexcept = default;

  // --------------------------------------------------------------------------
  // Accessors
  // --------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numVertices() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numFaces() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  face(Size i) const noexcept -> Face;

  // --------------------------------------------------------------------------
  // Methods
  // --------------------------------------------------------------------------

  PURE [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE [[nodiscard]] constexpr auto
  faceContaining(Point<D, T> const & p) const noexcept -> Size;

  // Create a RegularPartition with approximately numFaces() * mesh_multiplier cells.
  // For child index i, the faces that intersect that cell are at indices
  // children[i] to children[i+1] in the face_ids_buffer.
  // That is, children[i+1] - children[i] is the number of faces that intersect
  // the cell, and children[i] is the index of the first face that intersects
  // the cell in the face_ids_buffer.
  //
  // Input:
  // --------
  // face_ids_buffer: an empty vector
  // mesh_multiplier: the target number of cells per face
  //
  // Output:
  // --------
  // RegularPartition with approximately numFaces() * mesh_multiplier cells, whose
  // children are indices into face_ids_buffer. face_ids_buffer is filled with the indices
  // of the faces that intersect each cell.
  [[nodiscard]] constexpr auto
  regularPartition(Vector<I> & face_ids_buffer, T mesh_multiplier = 1) const noexcept
      -> RegularPartition<D, T, I>;
};

} // namespace um2

#include "TriMesh.inl"
