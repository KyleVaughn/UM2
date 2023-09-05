namespace um2
{

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
validateMesh(FaceVertexMesh<P, N, D, T, I> & mesh)
{
#ifndef NDEBUG
  // Check for repeated vertices.
  // This is not technically an error, but it is a sign that the mesh may
  // cause problems for some algorithms. Hence, we warn the user.
  auto const bbox = boundingBox(mesh);
  Vec<D, T> normalization;
  for (Size i = 0; i < D; ++i) {
    normalization[i] = static_cast<T>(1) / (bbox.maxima[i] - bbox.minima[i]);
  }
  Vector<Point<D, T>> vertices_copy = mesh.vertices;
  // Transform the points to be in the unit cube
  for (auto & v : vertices_copy) {
    v -= bbox.minima;
    v *= normalization;
  }
  if constexpr (std::same_as<T, float>) {
    mortonSort<uint32_t>(vertices_copy.begin(), vertices_copy.end());
  } else {
    mortonSort<uint64_t>(vertices_copy.begin(), vertices_copy.end());
  }
  // Revert the scaling
  for (auto & v : vertices_copy) {
    // cppcheck-suppress useStlAlgorithm; justification: This is less verbose
    v /= normalization;
  }
  Size const num_vertices = mesh.numVertices();
  for (Size i = 0; i < num_vertices - 1; ++i) {
    if (isApprox(vertices_copy[i], vertices_copy[i + 1])) {
      Log::warn("Vertex " + std::to_string(i) + " and " + std::to_string(i + 1) +
                " are effectively equivalent");
    }
  }
#endif

  // Check that the vertices are in counter-clockwise order.
  // If the area of the face is negative, then the vertices are in clockwise
  Size const num_faces = mesh.numFaces();
  for (Size i = 0; i < num_faces; ++i) {
    if (!mesh.getFace(i).isCCW()) {
      Log::warn("Face " + std::to_string(i) +
                " has vertices in clockwise order. Reordering");
      mesh.flipFace(i);
    }
  }

  // Convexity check
  // if (file.type == MeshType::Quad) {
  if constexpr (N == 4) {
    for (Size i = 0; i < num_faces; ++i) {
      if (!isConvex(mesh.getFace(i))) {
        Log::warn("Face " + std::to_string(i) + " is not convex");
      }
    }
  }
}

} // namespace um2
