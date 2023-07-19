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
};

} // namespace um2
