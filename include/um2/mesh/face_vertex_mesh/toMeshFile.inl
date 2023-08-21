namespace um2
{

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
toMeshFile(FaceVertexMesh<P, N, D, T, I> const & mesh, MeshFile<T, I> & file) noexcept
{
  // Default to XDMf
  file.format = MeshFileFormat::XDMF;
  file.type = getMeshType<P, N>();

  // Vertices
  if constexpr (D == 3) {
    file.vertices = mesh.vertices;
  } else {
    file.vertices.resize(static_cast<size_t>(mesh.numVertices()));
    for (Size i = 0; i < mesh.numVertices(); ++i) {
      file.vertices[static_cast<size_t>(i)][0] = mesh.vertices[i][0];
      file.vertices[static_cast<size_t>(i)][1] = mesh.vertices[i][1];
      file.vertices[static_cast<size_t>(i)][2] = 0;
    }
  }

  // Faces
  // NOLINTBEGIN(bugprone-misplaced-widening-cast) justification: It's not misplaced...
  auto const len = static_cast<size_t>(mesh.numFaces() * N);
  file.element_conn.resize(len);
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    for (Size j = 0; j < N; ++j) {
      file.element_conn[static_cast<size_t>(i * N + j)] = mesh.fv[i][j];
    }
  }
  // NOLINTEND(bugprone-misplaced-widening-cast)
}

} // namespace um2
