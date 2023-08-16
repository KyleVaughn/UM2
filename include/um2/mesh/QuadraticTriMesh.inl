namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
QuadraticTriMesh<D, T, I>::FaceVertexMesh(MeshFile<T, I> const & file)
{
  assert(!file.vertices.empty());
  assert(!file.element_conn.empty());

  auto const num_vertices = static_cast<Size>(file.vertices.size());
  auto const num_faces = static_cast<Size>(file.numCells());
  auto const conn_size = static_cast<Size>(file.element_conn.size());
  auto const verts_per_face = verticesPerCell(file.type);
  if (verts_per_face != 6) {
    Log::error(
        "Attempted to construct a QuadraticTriMesh from a mesh file with non-tri6 faces");
  }
  assert(conn_size == num_faces * verts_per_face);

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
    vertices.resize(num_vertices);
    for (Size i = 0; i < num_vertices; ++i) {
      vertices[i][0] = file.vertices[static_cast<size_t>(i)][0];
      vertices[i][1] = file.vertices[static_cast<size_t>(i)][1];
    }
  } else {
    vertices = file.vertices;
  }

  // -- Connectivity --
  fv.resize(num_faces);
  for (Size i = 0; i < num_faces; ++i) {
    for (Size j = 0; j < verts_per_face; ++j) {
      auto const idx = i * verts_per_face + j;
      fv[i][j] = file.element_conn[static_cast<size_t>(idx)];
    }
  }

  // -- Vertex/Face connectivity --
  Vector<I> vert_counts(num_vertices, 0);
  for (size_t i = 0; i < static_cast<size_t>(conn_size); ++i) {
    ++vert_counts[static_cast<Size>(file.element_conn[i])];
  }
  vf_offsets.resize(num_vertices + 1);
  vf_offsets[0] = 0;
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(), vf_offsets.begin() + 1);
  vert_counts.clear();
  vf.resize(static_cast<Size>(vf_offsets[num_vertices]));
  // Copy vf_offsets to vert_offsets
  Vector<I> vert_offsets = vf_offsets;
  for (Size i = 0; i < num_faces; ++i) {
    auto const & face = fv[i];
    for (Size j = 0; j < verts_per_face; ++j) {
      auto const vert = static_cast<Size>(face[j]);
      vf[static_cast<Size>(vert_offsets[vert])] = static_cast<I>(i);
      ++vert_offsets[vert];
    }
  }

#ifndef NDEBUG
  // Check for repeated vertices.
  // This is not technically an error, but it is a sign that the mesh may
  // cause problems for some algorithms. Hence, we warn the user.
  auto const bbox = boundingBox();
  Vec<D, T> normalization;
  for (Size i = 0; i < D; ++i) {
    normalization[i] = static_cast<T>(1) / (bbox.maxima[i] - bbox.minima[i]);
  }
  Vector<Point<D, T>> vertices_copy = this->vertices;
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
  for (Size i = 0; i < num_vertices - 1; ++i) {
    if (isApprox(vertices_copy[i], vertices_copy[i + 1])) {
      Log::warn("Vertex " + std::to_string(i) + " and " + std::to_string(i + 1) +
                " are effectively equivalent");
    }
  }

  // Check that the vertices are in counter-clockwise order.
  // If the area of the face is negative, then the vertices are in clockwise
  for (Size i = 0; i < num_faces; ++i) {
    auto & face = fv[i];
    I & vi0 = face[0];
    I & vi1 = face[1];
    I & vi2 = face[2];
    Point2<T> const & v0 = this->vertices[static_cast<Size>(vi0)];
    Point2<T> const & v1 = this->vertices[static_cast<Size>(vi1)];
    Point2<T> const & v2 = this->vertices[static_cast<Size>(vi2)];
    if (!Triangle2<T>(v0, v1, v2).isCCW()) {
      Log::warn("Face " + std::to_string(i) +
                " has vertices in clockwise order. Reordering");
      std::swap(vi1, vi2);
      std::swap(face[3], face[5]);
    }
  }
#endif
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadraticTriMesh<D, T, I>::numVertices() const noexcept -> Size
{
  return vertices.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadraticTriMesh<D, T, I>::numFaces() const noexcept -> Size
{
  return fv.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadraticTriMesh<D, T, I>::face(Size i) const noexcept -> Face
{
  auto const v0 = static_cast<Size>(fv[i][0]);
  auto const v1 = static_cast<Size>(fv[i][1]);
  auto const v2 = static_cast<Size>(fv[i][2]);
  auto const v3 = static_cast<Size>(fv[i][3]);
  auto const v4 = static_cast<Size>(fv[i][4]);
  auto const v5 = static_cast<Size>(fv[i][5]);
  return QuadraticTriangle<D, T>(vertices[v0], vertices[v1], vertices[v2], vertices[v3],
                                 vertices[v4], vertices[v5]);
}

// -------------------------------------------------------------------
// Methods
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
QuadraticTriMesh<D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  AxisAlignedBox<D, T> box = face(0).boundingBox();
  for (Size i = 1; i < numFaces(); ++i) {
    box = um2::boundingBox(box, face(i).boundingBox());
  }
  return box;
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
QuadraticTriMesh<D, T, I>::faceContaining(Point<D, T> const & p) const noexcept -> Size
{
  static_assert(D == 2, "Only implemented for 2D meshes");
  for (Size i = 0; i < numFaces(); ++i) {
    if (face(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1;
}

template <Size D, std::floating_point T, std::signed_integral I>
void
QuadraticTriMesh<D, T, I>::toMeshFile(MeshFile<T, I> & file) const noexcept
{
  // Default to XDMF
  file.format = MeshFileFormat::XDMF;
  file.type = MeshType::QuadraticTri;
  Size const n = 6;

  // Vertices
  if constexpr (D == 3) {
    file.vertices = vertices;
  } else {
    file.vertices.resize(static_cast<size_t>(numVertices()));
    for (Size i = 0; i < numVertices(); ++i) {
      file.vertices[static_cast<size_t>(i)][0] = vertices[i][0];
      file.vertices[static_cast<size_t>(i)][1] = vertices[i][1];
      file.vertices[static_cast<size_t>(i)][2] = 0;
    }
  }

  // Faces
  // NOLINTBEGIN(bugprone-misplaced-widening-cast)
  file.element_conn.resize(static_cast<size_t>(numFaces() * n));
  for (Size i = 0; i < numFaces(); ++i) {
    for (Size j = 0; j < n; ++j) {
      file.element_conn[static_cast<size_t>(i * n + j)] = fv[i][j];
    }
  }
  // NOLINTEND(bugprone-misplaced-widening-cast)
}

} // namespace um2
