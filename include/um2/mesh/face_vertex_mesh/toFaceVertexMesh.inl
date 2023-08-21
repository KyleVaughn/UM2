namespace um2
{

// Return true if the MeshType and P, N are compatible.
template <Size P, Size N>
constexpr auto
validateMeshFileType(MeshType const type) -> bool
{
  if constexpr (P == 1 && N == 3) {
    return type == MeshType::Tri;
  } else if constexpr (P == 1 && N == 4) {
    return type == MeshType::Quad;
  } else if constexpr (P == 2 && N == 6) {
    return type == MeshType::QuadraticTri;
  } else if constexpr (P == 2 && N == 8) {
    return type == MeshType::QuadraticQuad;
  }
  return false;
}

template <Size P, Size N>
constexpr auto
getMeshType() -> MeshType
{
  if constexpr (P == 1 && N == 3) {
    return MeshType::Tri;
  } else if constexpr (P == 1 && N == 4) {
    return MeshType::Quad;
  } else if constexpr (P == 2 && N == 6) {
    return MeshType::QuadraticTri;
  } else if constexpr (P == 2 && N == 8) {
    return MeshType::QuadraticQuad;
  }
  return MeshType::None;
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
toFaceVertexMesh(MeshFile<T, I> const & file,
                 FaceVertexMesh<P, N, D, T, I> & mesh) noexcept
{
  assert(!file.vertices.empty());
  assert(!file.element_conn.empty());
  auto const num_vertices = static_cast<Size>(file.vertices.size());
  auto const num_faces = static_cast<Size>(file.numCells());
  auto const conn_size = static_cast<Size>(file.element_conn.size());
  if (!validateMeshFileType<P, N>(file.type)) {
    Log::error("Attempted to construct a FaceVertexMesh from a mesh file with an "
               "incompatible mesh type");
  }
  assert(conn_size == num_faces * verticesPerCell(file.type));

  // -- Vertices --
  // Ensure each of the vertices has approximately the same z
  if constexpr (D == 2) {
#ifndef NDEBUG
    T const eps = epsilonDistance<T>();
    T const z = file.vertices[0][2];
    for (auto const & v : file.vertices) {
      assert(std::abs(v[2] - z) < eps);
    }
#endif
    mesh.vertices.resize(num_vertices);
    for (Size i = 0; i < num_vertices; ++i) {
      mesh.vertices[i][0] = file.vertices[static_cast<size_t>(i)][0];
      mesh.vertices[i][1] = file.vertices[static_cast<size_t>(i)][1];
    }
  } else {
    mesh.vertices = file.vertices;
  }

  // -- Face/Vertex connectivity --
  mesh.fv.resize(num_faces);
  for (Size i = 0; i < num_faces; ++i) {
    for (Size j = 0; j < N; ++j) {
      auto const idx = i * N + j;
      mesh.fv[i][j] = file.element_conn[static_cast<size_t>(idx)];
    }
  }

  // -- Vertex/Face connectivity --
  Vector<I> vert_counts(num_vertices, 0);
  for (size_t i = 0; i < static_cast<size_t>(conn_size); ++i) {
    ++vert_counts[static_cast<Size>(file.element_conn[i])];
  }
  mesh.vf_offsets.resize(num_vertices + 1);
  mesh.vf_offsets[0] = 0;
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(),
                      mesh.vf_offsets.begin() + 1);
  vert_counts.clear();
  mesh.vf.resize(static_cast<Size>(mesh.vf_offsets[num_vertices]));
  // Copy vf_offsets to vert_offsets
  Vector<I> vert_offsets = mesh.vf_offsets;
  for (Size i = 0; i < num_faces; ++i) {
    auto const & face = mesh.fv[i];
    for (Size j = 0; j < N; ++j) {
      auto const vert = static_cast<Size>(face[j]);
      mesh.vf[static_cast<Size>(vert_offsets[vert])] = static_cast<I>(i);
      ++vert_offsets[vert];
    }
  }
  validateMesh(mesh);
}

} // namespace um2
