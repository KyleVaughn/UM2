namespace um2
{

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
toMeshFile(FaceVertexMesh<P, N, D, T, I> const & mesh, MeshFile<T, I> & file) noexcept
{
  // Default to XDMf
  file.format = MeshFileFormat::XDMF;

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
  auto const nfaces = static_cast<size_t>(mesh.numFaces());
  auto const n = static_cast<size_t>(N);
  auto const len = nfaces * n;
  MeshType const mesh_type = getMeshType<P, N>();
  file.element_types.resize(nfaces, mesh_type);
  file.element_offsets.resize(nfaces + 1U);
  file.element_conn.resize(len);
  for (size_t i = 0; i < nfaces; ++i) {
    file.element_offsets[i] = static_cast<I>(i * n);
    for (size_t j = 0; j < n; ++j) {
      file.element_conn[i * n + j] = mesh.fv[static_cast<Size>(i)][static_cast<Size>(j)];
    }
  }
  file.element_offsets[nfaces] = static_cast<I>(len);
  // NOLINTEND(bugprone-misplaced-widening-cast)
}

} // namespace um2
