#pragma once

#include <um2/geometry/Quadrilateral.hpp>
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/mesh/FaceVertexMesh.hpp>
#include <um2/mesh/MeshFile.hpp>
#include <um2/mesh/RegularPartition.hpp>

namespace um2
{

template <Size D, std::floating_point T, std::signed_integral I>
struct FaceVertexMesh<1, 4, D, T, I> {

  using FaceConn = Vec<4, I>;
  using Face = Quadrilateral<D, T>;

  Vector<Point<D, T>> vertices;
  Vector<FaceConn> fv;
  Vector<I> vf_offsets; // size = num_vertices + 1
  Vector<I> vf;         // size = vf_offsets[num_vertices]

  // --------------------------------------------------------------------------
  // Constructors
  // --------------------------------------------------------------------------

  constexpr FaceVertexMesh() noexcept = default;

  explicit FaceVertexMesh(MeshFile<T, I> const & file);

  // --------------------------------------------------------------------------
  // Accessors
  // --------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numVertices() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numFaces() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFace(Size i) const noexcept -> Face;

  // --------------------------------------------------------------------------
  // Methods
  // --------------------------------------------------------------------------

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

#include "QuadMesh.inl"