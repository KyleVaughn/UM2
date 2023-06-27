#pragma once

#include <um2/mesh/face_vertex_mesh.hpp>

namespace um2
{

template <std::floating_point T, std::signed_integral I>
struct FaceVertexMesh<1, 3, T, I> {
  Vector<Point2<T>> vertices;
  Vector<I> fv_offsets; // size = num_faces + 1
  Vector<I> fv;         // size = fv_offsets[num_faces]
  Vector<I> vf_offsets; // size = num_vertices + 1
  Vector<I> vf;         // size = vf_offsets[num_vertices]
  // -- Constructors --
  UM2_HOSTDEV
  FaceVertexMesh() = default;
  //  FaceVertexMesh(MeshFile<T, I> const &);
  // -- Methods --
  //  void to_mesh_file(MeshFile<T, I> &) const;
};

} // namespace um2