#pragma once

#include <um2/mesh/FaceVertexMesh.hpp>

#include <um2/geometry/Triangle.hpp>

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

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  faceContaining(Point<D, T> const & p) const noexcept -> Size;

};

} // namespace um2

#include "TriMesh.inl"
