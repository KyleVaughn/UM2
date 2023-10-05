// Free functions
namespace um2
{

//==============================================================================
//==============================================================================
// Free functions
//==============================================================================
//==============================================================================

//==============================================================================
// numVertices
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numVertices(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.vertices.size();
}

//==============================================================================
// numFaces
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numFaces(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.fv.size();
}

//==============================================================================
// TriMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(TriMesh<D, T, I> const & mesh, Size i) noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                        mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                        mesh.vertices[static_cast<Size>(mesh.fv[i][2])]);
}

//==============================================================================
// QuadMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadMesh<D, T, I> const & mesh, Size i) noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][3])]);
}

//==============================================================================
// QuadraticTriMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadraticTriMesh<D, T, I> const & mesh, Size i) noexcept
    -> QuadraticTriangle<D, T>
{
  return QuadraticTriangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][3])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][4])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][5])]);
}

//==============================================================================
// QuadraticQuadMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadraticQuadMesh<D, T, I> const & mesh, Size i) noexcept
    -> QuadraticQuadrilateral<D, T>
{
  return QuadraticQuadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][3])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][4])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][5])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][6])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][7])]);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size N, Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
boundingBox(LinearPolygonMesh<N, D, T, I> const & mesh) noexcept -> AxisAlignedBox<D, T>
{
  return boundingBox(mesh.vertices);
}

template <Size N, Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
boundingBox(QuadraticPolygonMesh<N, D, T, I> const & mesh) noexcept
    -> AxisAlignedBox<D, T>
{
  AxisAlignedBox<D, T> box = mesh.getFace(0).boundingBox();
  for (Size i = 1; i < numFaces(mesh); ++i) {
    box += mesh.getFace(i).boundingBox();
  }
  return box;
}

//==============================================================================
// faceContaining
//==============================================================================

template <Size P, Size N, std::floating_point T, std::signed_integral I>
PURE constexpr auto
faceContaining(PlanarPolygonMesh<P, N, T, I> const & mesh, Point2<T> const & p) noexcept
    -> Size
{
  for (Size i = 0; i < numFaces(mesh); ++i) {
    if (mesh.getFace(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1;
}

//==============================================================================
// validateMesh
//==============================================================================

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

//==============================================================================
// toFaceVertexMesh
//==============================================================================

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
  MeshType const meshtype = file.getMeshType();
  if (!validateMeshFileType<P, N>(meshtype)) {
    Log::error("Attempted to construct a FaceVertexMesh from a mesh file with an "
               "incompatible mesh type");
  }
  assert(conn_size == num_faces * verticesPerCell(meshtype));

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

//==============================================================================
// toMeshFile
//==============================================================================

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

//==============================================================================
// intersect
//==============================================================================

// template <Size N, std::floating_point T, std::signed_integral I>
// void
// intersect(PlanarLinearPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
//           T * intersections, Size * const n) noexcept
//{
//   T const r_miss = infiniteDistance<T>();
//   // cppcheck-suppress constStatement; justification: false positive
//   T * const start = intersections;
// #ifndef NDEBUG
//   Size const n0 = *n;
// #endif
//   for (Size i = 0; i < numFaces(mesh); ++i) {
//     auto const face = mesh.getFace(i);
//     for (Size j = 0; j < polygonNumEdges<1, N>(); ++j) {
//       auto const edge = face.getEdge(j);
//       T const r = intersect(edge, ray);
//       if (r < r_miss) {
//         assert(static_cast<Size>(intersections - start) < n0);
//         *intersections++ = r;
//       }
//     }
//   }
//   *n = static_cast<Size>(intersections - start);
//   std::sort(start, intersections);
// }

template <Size N, std::floating_point T, std::signed_integral I>
void
intersect(PlanarLinearPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
          T * const intersections, Size * const n) noexcept
{
  T constexpr r_miss = infiniteDistance<T>();
  Size nintersect = 0;
#ifndef NDEBUG
  Size const n0 = *n;
#endif
  Size constexpr edges_per_face = PlanarLinearPolygonMesh<N, T, I>::Face::numEdges();
  for (Size i = 0; i < numFaces(mesh); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < edges_per_face; ++j) {
      auto const edge = face.getEdge(j);
      T const r = intersect(edge, ray);
      if (r < r_miss) {
        assert(nintersect < n0);
        intersections[nintersect++] = r;
      }
    }
  }
  *n = nintersect;
  std::sort(intersections, intersections + nintersect);
}

template <Size N, std::floating_point T, std::signed_integral I>
void
intersect(PlanarQuadraticPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
          T * const intersections, Size * const n) noexcept
{
  T constexpr r_miss = infiniteDistance<T>();
  Size nintersect = 0;
#ifndef NDEBUG
  Size const n0 = *n;
#endif
  Size constexpr edges_per_face = PlanarQuadraticPolygonMesh<N, T, I>::Face::numEdges();
  for (Size i = 0; i < numFaces(mesh); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < edges_per_face; ++j) {
      auto const edge = face.getEdge(j);
      auto const r = intersect(edge, ray);
      if (r[0] < r_miss) {
        assert(nintersect < n0);
        intersections[nintersect++] = r[0];
      }
      if (r[1] < r_miss) {
        assert(nintersect < n0);
        intersections[nintersect++] = r[1];
      }
    }
  }
  *n = nintersect;
  std::sort(intersections, intersections + nintersect);
}

//==============================================================================
// getUniqueEdges
//==============================================================================

template <Size N, Size D, std::floating_point T, std::signed_integral I>
void
getUniqueEdges(LinearPolygonMesh<N, D, T, I> const & mesh,
               Vector<LineSegment<D, T>> & edges)
{
  Size const edges_per_face = N;
  Size const nfaces = numFaces(mesh);
  Size const non_unique_edges = nfaces * edges_per_face;
  Vector<Vec2<I>> edge_conn_vec(non_unique_edges);
  for (Size iface = 0; iface < nfaces; ++iface) {
    auto const & face_conn = mesh.fv[iface];
    for (Size iedge = 0; iedge < edges_per_face - 1; ++iedge) {
      Vec2<I> edge_conn(face_conn[iedge], face_conn[iedge + 1]);
      if (edge_conn[0] > edge_conn[1]) {
        std::swap(edge_conn[0], edge_conn[1]);
      }
      edge_conn_vec[iface * edges_per_face + iedge] = edge_conn;
    }
    Vec2<I> edge_conn(face_conn[edges_per_face - 1], face_conn[0]);
    if (edge_conn[0] > edge_conn[1]) {
      std::swap(edge_conn[0], edge_conn[1]);
    }
    edge_conn_vec[iface * edges_per_face + edges_per_face - 1] = edge_conn;
  }

  // Sort the edges by the first vertex index, then the second vertex index.
  std::sort(edge_conn_vec.begin(), edge_conn_vec.end(),
            [](auto const & a, auto const & b) {
              return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
            });

  // Get the unique edges.
  auto last = std::unique(edge_conn_vec.begin(), edge_conn_vec.end());
  Size const nunique_edges = static_cast<Size>(last - edge_conn_vec.begin());

  // Copy the unique edges into the output vector.
  edges.resize(nunique_edges);
  for (Size i = 0; i < nunique_edges; ++i) {
    auto const edge_conn = edge_conn_vec[i];
    edges[i] =
        LineSegment<D, T>(mesh.vertices[edge_conn[0]], mesh.vertices[edge_conn[1]]);
  }
}

template <Size N, Size D, std::floating_point T, std::signed_integral I>
void
getUniqueEdges(QuadraticPolygonMesh<N, D, T, I> const & mesh,
               Vector<QuadraticSegment<D, T>> & edges)
{
  Size const edges_per_face = N / 2;
  Size const nfaces = numFaces(mesh);
  Size const non_unique_edges = nfaces * edges_per_face;
  Vector<Vec3<I>> edge_conn_vec(non_unique_edges);
  for (Size iface = 0; iface < nfaces; ++iface) {
    auto const & face_conn = mesh.fv[iface];
    for (Size iedge = 0; iedge < edges_per_face - 1; ++iedge) {
      Vec3<I> edge_conn(face_conn[iedge], face_conn[iedge + 1],
                        face_conn[iedge + edges_per_face]);
      if (edge_conn[0] > edge_conn[1]) {
        std::swap(edge_conn[0], edge_conn[1]);
      }
      edge_conn_vec[iface * edges_per_face + iedge] = edge_conn;
    }
    Vec3<I> edge_conn(face_conn[edges_per_face - 1], face_conn[0], face_conn[N - 1]);
    if (edge_conn[0] > edge_conn[1]) {
      std::swap(edge_conn[0], edge_conn[1]);
    }
    edge_conn_vec[iface * edges_per_face + edges_per_face - 1] = edge_conn;
  }

  // Sort the edges by the first vertex index, then the second vertex index.
  std::sort(edge_conn_vec.begin(), edge_conn_vec.end(),
            [](auto const & a, auto const & b) {
              return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]) ||
                     (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]);
            });

  // Get the unique edges.
  auto last = std::unique(edge_conn_vec.begin(), edge_conn_vec.end());
  Size const nunique_edges = static_cast<Size>(last - edge_conn_vec.begin());

  // Copy the unique edges into the output vector.
  edges.resize(nunique_edges);
  for (Size i = 0; i < nunique_edges; ++i) {
    auto const edge_conn = edge_conn_vec[i];
    edges[i] =
        QuadraticSegment<D, T>(mesh.vertices[edge_conn[0]], mesh.vertices[edge_conn[1]],
                               mesh.vertices[edge_conn[2]]);
  }
}

//==============================================================================
//==============================================================================
// Member functions
//==============================================================================
//==============================================================================

//==============================================================================
// Constructors
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
FaceVertexMesh<P, N, D, T, I>::FaceVertexMesh(MeshFile<T, I> const & file)
{
  um2::toFaceVertexMesh(file, *this);
}

//==============================================================================
// numVertices
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numVertices() const noexcept -> Size
{
  return um2::numVertices(*this);
}

//==============================================================================
// numFaces
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numFaces() const noexcept -> Size
{
  return um2::numFaces(*this);
}

//==============================================================================
// getFace
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::getFace(Size i) const noexcept -> Face
{
  return um2::getFace(*this, i);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// faceContaining
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::faceContaining(Point<D, T> const & p) const noexcept
    -> Size
  requires(D == 2)
{
  return um2::faceContaining(*this, p);
}

//==============================================================================
// flipFace
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::flipFace(Size i) noexcept
{
  if constexpr (P == 1 && N == 3) {
    um2::swap(fv[i][1], fv[i][2]);
  } else if constexpr (P == 1 && N == 4) {
    um2::swap(fv[i][1], fv[i][3]);
  } else if constexpr (P == 2 && N == 6) {
    um2::swap(fv[i][1], fv[i][2]);
    um2::swap(fv[i][3], fv[i][5]);
  } else if constexpr (P == 2 && N == 8) {
    um2::swap(fv[i][1], fv[i][3]);
    um2::swap(fv[i][4], fv[i][7]);
  }
}

//==============================================================================
// toMeshFile
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::toMeshFile(MeshFile<T, I> & file) const noexcept
{
  um2::toMeshFile(*this, file);
}

//==============================================================================
// getFaceAreas
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::getFaceAreas(Vector<T> & areas) const noexcept
{
  areas.resize(numFaces());
  for (Size i = 0; i < numFaces(); ++i) {
    areas[i] = getFace(i).area();
  }
}

//==============================================================================
// getUniqueEdges
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::getUniqueEdges(Vector<Edge> & edges) const noexcept
{
  um2::getUniqueEdges(*this, edges);
}

//==============================================================================
// intersect
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::intersect(Ray<D, T> const & ray, T * intersections,
                                         Size * const n) const noexcept
  requires(D == 2)
{
  um2::intersect(*this, ray, intersections, n);
}

//==============================================================================
// printStats
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
printStats(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept
{
  using Edge = typename FaceVertexMesh<P, N, D, T, I>::Edge;
  Vector<T> data;
  std::vector<T> std_data;

  // num faces & vertices
  std::cout << "num faces: " << mesh.numFaces() << '\n';
  std::cout << "num vertices: " << mesh.numVertices() << '\n';

  // Face areas
  mesh.getFaceAreas(data);
  std_data.resize(static_cast<size_t>(data.size()));
  std::copy(data.begin(), data.end(), std_data.begin());
  std::sort(std_data.begin(), std_data.end());
  std::cout << "\nFace areas:\n";
  printHistogram(std_data);

  // Edge lengths
  Vector<Edge> unique_edges;
  mesh.getUniqueEdges(unique_edges);
  data.resize(unique_edges.size());
  for (Size i = 0; i < unique_edges.size(); ++i) {
    data[i] = unique_edges[i].length();
  }
  std_data.resize(static_cast<size_t>(data.size()));
  std::copy(data.begin(), data.end(), std_data.begin());
  std::sort(std_data.begin(), std_data.end());
  std::cout << "\nEdge lengths:\n";
  printHistogram(std_data);

  // Mean chord length
  data.resize(mesh.numFaces());
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    data[i] = mesh.getFace(i).meanChordLength();
  }
  std_data.resize(static_cast<size_t>(data.size()));
  std::copy(data.begin(), data.end(), std_data.begin());
  std::sort(std_data.begin(), std_data.end());
  std::cout << "\nMean chord lengths:\n";
  printHistogram(std_data);
}

} // namespace um2
