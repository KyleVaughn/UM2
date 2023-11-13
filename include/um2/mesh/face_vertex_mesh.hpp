#pragma once

#include <um2/mesh/polytope_soup.hpp>

namespace um2
{

//=============================================================================
// FACE-VERTEX MESH
//=============================================================================
//
// A 2D volumetric or 3D surface mesh composed of polygons of polynomial order P.
// Each polygon (face) is composed of N vertices. Each vertex is a D-dimensional
// point of floating point type T.
//  - P = 1, N = 3: Triangular mesh
//  - P = 1, N = 4: Quadrilateral mesh
//  - P = 2, N = 6: Quadratic triangular mesh
//  - P = 2, N = 8: Quadratic quadrilateral mesh
//
// Let I be the signed integer type used to index vertices and faces.
// We will use some simple meshes to explain the data structure. A more detailed
// explanation of each member follows.
//  - A TriMesh (FaceVertexMesh<1, 3>) with two triangles:
//      3---2
//      | / |
//      0---1
//      vertices = { {0, 0}, {1, 0}, {1, 1}, {0, 1} }
//          4 vertices on the unit square
//      fv = { {0, 1, 2}, {2, 3, 0} }
//          The 6 vertex indices composing the two triangles {0, 1, 2} and {2, 3, 0}
//      vf = { 0, 1, 0, 0, 1, 1 }
//          The face indices to which each vertex belongs. More precisely, vertex
//          0 belongs to faces 0 and 1, vertex 1 belongs to face 0 only, etc.
//          Face IDs are ordered least to greatest.
//      vf_offsets = { 0, 2, 3, 5, 6}
//          vf_offsets[i] is the index of the smallest face ID to which vertex i
//          belongs. There is an additional element at the end, which is the length
//          of the vf vector. Used to calculate the number of faces to which each
//          vertex belongs.
//
template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
struct FaceVertexMesh {

  using FaceConn = Vec<N, I>;
  using Face = Polygon<P, N, D, T>;
  using Edge = typename Polygon<P, N, D, T>::Edge;

  Vector<Point<D, T>> vertices;
  Vector<FaceConn> fv;
  Vector<I> vf_offsets; // size = num_vertices + 1
  Vector<I> vf;         // size = vf_offsets[num_vertices]

  //===========================================================================
  // Constructors
  //===========================================================================

  constexpr FaceVertexMesh() noexcept = default;

  explicit FaceVertexMesh(PolytopeSoup<T, I> const & soup);

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numVertices() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numFaces() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFace(Size i) const noexcept -> Face;

  //===========================================================================
  // Methods
  //===========================================================================

  PURE [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D, T>;

  PURE [[nodiscard]] constexpr auto
  faceContaining(Point<D, T> const & p) const noexcept -> Size
    requires(D == 2);

  void
  flipFace(Size i) noexcept;

//  void
//  toPolytopeSoup(PolytopeSoup<T, I> & soup) const noexcept;
//
//  //  void
//  //  getFaceAreas(Vector<T> & areas) const noexcept;
//  //
//  //  void
//  //  getUniqueEdges(Vector<Edge> & edges) const noexcept;
//  //
//  void
//  intersect(Ray<D, T> const & ray, T * intersections, Size * n) const noexcept
//    requires(D == 2);
};

//==============================================================================
// Aliases
//==============================================================================

// Polynomial order
template <Size N, Size D, std::floating_point T, std::signed_integral I>
using LinearPolygonMesh = FaceVertexMesh<1, N, D, T, I>;
template <Size N, Size D, std::floating_point T, std::signed_integral I>
using QuadraticPolygonMesh = FaceVertexMesh<2, N, D, T, I>;

// Number of vertices per face
template <Size D, std::floating_point T, std::signed_integral I>
using TriMesh = LinearPolygonMesh<3, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using QuadMesh = LinearPolygonMesh<4, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticTriMesh = QuadraticPolygonMesh<6, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticQuadMesh = QuadraticPolygonMesh<8, D, T, I>;

// 2D
template <Size P, Size N, std::floating_point T, std::signed_integral I>
using PlanarPolygonMesh = FaceVertexMesh<P, N, 2, T, I>;
template <Size N, std::floating_point T, std::signed_integral I>
using PlanarLinearPolygonMesh = FaceVertexMesh<1, N, 2, T, I>;
template <Size N, std::floating_point T, std::signed_integral I>
using PlanarQuadraticPolygonMesh = FaceVertexMesh<2, N, 2, T, I>;

//==============================================================================
// numVertices
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numVertices(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.vertices.size();
}

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
PURE HOSTDEV constexpr auto
numFaces(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.fv.size();
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numFaces() const noexcept -> Size
{
  return um2::numFaces(*this);
}

//==============================================================================
// getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(TriMesh<D, T, I> const & mesh, Size i) noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                        mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                        mesh.vertices[static_cast<Size>(mesh.fv[i][2])]);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadMesh<D, T, I> const & mesh, Size i) noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][3])]);
}

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

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::getFace(Size i) const noexcept -> Face
{
  return um2::getFace(*this, i);
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

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
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
  ASSERT(false);
  return -1;
}

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
// validateMesh
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
validateMesh(FaceVertexMesh<P, N, D, T, I> & mesh)
{
//#ifndef NDEBUG
//  // Check for repeated vertices.
//  // This is not technically an error, but it is a sign that the mesh may
//  // cause problems for some algorithms. Hence, we warn the user.
//  auto const bbox = boundingBox(mesh);
//  Vec<D, T> normalization;
//  for (Size i = 0; i < D; ++i) {
//    normalization[i] = static_cast<T>(1) / (bbox.maxima[i] - bbox.minima[i]);
//  }
//  Vector<Point<D, T>> vertices_copy = mesh.vertices;
//  // Transform the points to be in the unit cube
//  for (auto & v : vertices_copy) {
//    v -= bbox.minima;
//    v *= normalization;
//  }
//  if constexpr (std::same_as<T, float>) {
//    mortonSort<uint32_t>(vertices_copy.begin(), vertices_copy.end());
//  } else {
//    mortonSort<uint64_t>(vertices_copy.begin(), vertices_copy.end());
//  }
//  // Revert the scaling
//  for (auto & v : vertices_copy) {
//    // cppcheck-suppress useStlAlgorithm; justification: This is less verbose
//    v /= normalization;
//  }
//  Size const num_vertices = mesh.numVertices();
//  for (Size i = 0; i < num_vertices - 1; ++i) {
//    if (isApprox(vertices_copy[i], vertices_copy[i + 1])) {
//      Log::warn("Vertex " + toString(i) + " and " + toString(i + 1) +
//                " are effectively equivalent");
//    }
//  }
//#endif

  // Check that the vertices are in counter-clockwise order.
  // If the area of the face is negative, then the vertices are in clockwise
  Size const num_faces = mesh.numFaces();
  for (Size i = 0; i < num_faces; ++i) {
    if (!mesh.getFace(i).isCCW()) {
      Log::warn("Face " + toString(i) + " has vertices in clockwise order. Reordering");
      mesh.flipFace(i);
    }
  }

  // Convexity check
  if constexpr (N == 4) {
    for (Size i = 0; i < num_faces; ++i) {
      if (!isConvex(mesh.getFace(i))) {
        Log::warn("Face " + toString(i) + " is not convex");
      }
    }
  }
}

////==============================================================================
//// toFaceVertexMesh
////==============================================================================
//
// Return true if the MeshType and P, N are compatible.
template <Size P, Size N>
constexpr auto
validateMeshType(MeshType const type) -> bool
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
  ASSERT(false);
  return false;
}

//template <Size P, Size N>
//constexpr auto
//getMeshType() -> MeshType
//{
//  if constexpr (P == 1 && N == 3) {
//    return MeshType::Tri;
//  } else if constexpr (P == 1 && N == 4) {
//    return MeshType::Quad;
//  } else if constexpr (P == 2 && N == 6) {
//    return MeshType::QuadraticTri;
//  } else if constexpr (P == 2 && N == 8) {
//    return MeshType::QuadraticQuad;
//  }
//  ASSERT(false);
//  return MeshType::None;
//}
//
//template <Size P, Size N>
//constexpr auto
//getVTKElemType() -> VTKElemType
//{
//  if constexpr (P == 1 && N == 3) {
//    return VTKElemType::Triangle;
//  } else if constexpr (P == 1 && N == 4) {
//    return VTKElemType::Quad;
//  } else if constexpr (P == 2 && N == 6) {
//    return VTKElemType::QuadraticTriangle;
//  } else if constexpr (P == 2 && N == 8) {
//    return VTKElemType::QuadraticQuad;
//  }
//  ASSERT(false);
//  return VTKElemType::None;
//}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
toFaceVertexMesh(PolytopeSoup<T, I> const & soup,
                 FaceVertexMesh<P, N, D, T, I> & mesh) noexcept
{
  auto const num_vertices = soup.numVerts();
  auto const num_faces = soup.numElems();
  ASSERT(num_vertices != 0);
  ASSERT(num_faces != 0);
  MeshType const meshtype = soup.getMeshType();
  if (!validateMeshType<P, N>(meshtype)) {
    Log::error("Attempted to construct a FaceVertexMesh from an incompatible mesh type");
  }

  // -- Vertices --
  // Ensure each of the vertices has approximately the same z
  if constexpr (D == 2) {
#ifndef NDEBUG
    T const eps = eps_distance<T>;
    T const z = soup.getVertex(0)[2];
    for (Size i = 1; i < num_vertices; ++i) {
      ASSERT(um2::abs(soup.getVertex(i)[2] - z) < eps);
    }
#endif
    mesh.vertices.resize(num_vertices);
    for (Size i = 0; i < num_vertices; ++i) {
      auto const & p = soup.getVertex(i);
      mesh.vertices[i][0] = p[0];
      mesh.vertices[i][1] = p[1];
    }
  } else {
    for (Size i = 0; i < num_vertices; ++i) {
      auto const & p = soup.getVertex(i);
      mesh.vertices[i][0] = p[0];
      mesh.vertices[i][1] = p[1];
      mesh.vertices[i][2] = p[2];
    }
  }

  // -- Face/Vertex connectivity --
  Vector<I> conn(N);
  VTKElemType elem_type = VTKElemType::None;
  mesh.fv.resize(num_faces);
  for (Size i = 0; i < num_faces; ++i) {
    soup.getElement(i, elem_type, conn);
    for (Size j = 0; j < N; ++j) {
      mesh.fv[i][j] = conn[j];
    }
  }

  // -- Vertex/Face connectivity --
  Vector<I> vert_counts(num_vertices, 0);
  for (Size i = 0; i < num_faces; ++i) {
    for (Size j = 0; j < N; ++j) {
      ++vert_counts[static_cast<Size>(mesh.fv[i][j])];
    }
  }
  mesh.vf_offsets.resize(num_vertices + 1);
  mesh.vf_offsets[0] = 0;
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(),
                      mesh.vf_offsets.begin() + 1);
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

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
FaceVertexMesh<P, N, D, T, I>::FaceVertexMesh(PolytopeSoup<T, I> const & soup)
{
  um2::toFaceVertexMesh(soup, *this);
}

////==============================================================================
//// toPolytopeSoup
////==============================================================================
//
//template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
//void
//toPolytopeSoup(FaceVertexMesh<P, N, D, T, I> const & mesh,
//               PolytopeSoup<T, I> & soup) noexcept
//{
//  // Vertices
//  if constexpr (D == 3) {
//    soup.vertices = mesh.vertices;
//  } else {
//    soup.vertices.resize(mesh.numVertices());
//    for (Size i = 0; i < mesh.numVertices(); ++i) {
//      soup.vertices[i][0] = mesh.vertices[i][0];
//      soup.vertices[i][1] = mesh.vertices[i][1];
//      soup.vertices[i][2] = 0;
//    }
//  }
//
//  // Faces
//  auto const nfaces = mesh.numFaces();
//  auto const len = nfaces * N;
//  VTKElemType const elem_type = getVTKElemType<P, N>();
//  soup.element_types.resize(nfaces);
//  um2::fill(soup.element_types.begin(), soup.element_types.end(), elem_type);
//  soup.element_offsets.resize(nfaces + 1);
//  soup.element_conn.resize(len);
//  for (Size i = 0; i < nfaces; ++i) {
//    soup.element_offsets[i] = static_cast<I>(i * N);
//    for (Size j = 0; j < N; ++j) {
//      soup.element_conn[i * N + j] = mesh.fv[i][j];
//    }
//  }
//  soup.element_offsets[nfaces] = static_cast<I>(len);
//}
//
//template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
//void
//FaceVertexMesh<P, N, D, T, I>::toPolytopeSoup(PolytopeSoup<T, I> & soup) const noexcept
//{
//  um2::toPolytopeSoup(*this, soup);
//}
//
////==============================================================================
//// intersect
////==============================================================================
//
//template <Size N, std::floating_point T, std::signed_integral I>
//void
//intersect(PlanarLinearPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
//          T * const intersections, Size * const n) noexcept
//{
//  T constexpr r_miss = inf_distance<T>;
//  Size nintersect = 0;
//#ifndef NDEBUG
//  Size const n0 = *n;
//#endif
//  Size constexpr edges_per_face = PlanarLinearPolygonMesh<N, T, I>::Face::numEdges();
//  for (Size i = 0; i < numFaces(mesh); ++i) {
//    auto const face = mesh.getFace(i);
//    for (Size j = 0; j < edges_per_face; ++j) {
//      auto const edge = face.getEdge(j);
//      T const r = intersect(edge, ray);
//      if (r < r_miss) {
//        ASSERT(nintersect < n0);
//        intersections[nintersect++] = r;
//      }
//    }
//  }
//  *n = nintersect;
//  std::sort(intersections, intersections + nintersect);
//}
//
//template <Size N, std::floating_point T, std::signed_integral I>
//void
//intersect(PlanarQuadraticPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
//          T * const intersections, Size * const n) noexcept
//{
//  T constexpr r_miss = inf_distance<T>;
//  Size nintersect = 0;
//#ifndef NDEBUG
//  Size const n0 = *n;
//#endif
//  Size constexpr edges_per_face = PlanarQuadraticPolygonMesh<N, T, I>::Face::numEdges();
//  for (Size i = 0; i < numFaces(mesh); ++i) {
//    auto const face = mesh.getFace(i);
//    for (Size j = 0; j < edges_per_face; ++j) {
//      auto const edge = face.getEdge(j);
//      auto const r = intersect(edge, ray);
//      if (r[0] < r_miss) {
//        ASSERT(nintersect < n0);
//        intersections[nintersect++] = r[0];
//      }
//      if (r[1] < r_miss) {
//        ASSERT(nintersect < n0);
//        intersections[nintersect++] = r[1];
//      }
//    }
//  }
//  *n = nintersect;
//  std::sort(intersections, intersections + nintersect);
//}
//
//template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
//void
//FaceVertexMesh<P, N, D, T, I>::intersect(Ray<D, T> const & ray, T * intersections,
//                                         Size * const n) const noexcept
//  requires(D == 2)
//{
//  um2::intersect(*this, ray, intersections, n);
//}

} // namespace um2
