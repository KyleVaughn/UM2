#pragma once

#include <um2/config.hpp>

#include <um2/geometry/Point.hpp>
#include <um2/geometry/AxisAlignedBox.hpp>
#include <um2/stdlib/Vector.hpp>
#include <um2/mesh/MeshFile.hpp>

namespace um2
{

// FACE-VERTEX MESH
//-----------------------------------------------------------------------------
// A 2D volumetric or 3D surface mesh composed of polygons of polynomial order P.
// Each polygon (face) is composed of N vertices. Each vertex is a D-dimensional
// point of floating point type T.
//  - P = 1, N =  3: Triangular mesh
//  - P = 1, N =  4: Quadrilateral mesh
//  - P = 2, N =  6: Quadratic triangular mesh
//  - P = 2, N =  8: Quadratic quadrilateral mesh
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

  Vector<Point<D, T>> vertices;    
  Vector<FaceConn> fv;    
  Vector<I> vf_offsets; // size = num_vertices + 1    
  Vector<I> vf;         // size = vf_offsets[num_vertices]
};

// -----------------------------------------------------------------------------
// Aliases
// -----------------------------------------------------------------------------

template <Size N, Size D, std::floating_point T, std::signed_integral I>
using LinearPolygonMesh = FaceVertexMesh<1, N, D, T, I>;

template <Size N, Size D, std::floating_point T, std::signed_integral I>
using QuadraticPolygonMesh = FaceVertexMesh<2, N, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using TriMesh = LinearPolygonMesh<3, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadMesh = LinearPolygonMesh<4, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticTriMesh = QuadraticPolygonMesh<6, D, T, I>;

template <Size D, std::floating_point T, std::signed_integral I>
using QuadraticQuadMesh = QuadraticPolygonMesh<8, D, T, I>;

template <Size P, Size N, std::floating_point T, std::signed_integral I>
using PlanarPolygonMesh = FaceVertexMesh<P, N, 2, T, I>;

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------
// For all FaceVertexMesh, we define:
//   numVertices
//   numFaces
//   getFace
//   boundingBox
//   faceContaining(Point)
//   toFaceVertexMesh(MeshFile)
//   toMeshFile(FaceVertexMesh)

// -----------------------------------------------------------------------------
// numVertices
// -----------------------------------------------------------------------------
template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numVertices(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.vertices.size();
}

// -----------------------------------------------------------------------------
// numFaces
// -----------------------------------------------------------------------------
template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numFaces(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.fv.size();
}

// -----------------------------------------------------------------------------
// boundingBox
// -----------------------------------------------------------------------------
template <Size N, Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
boundingBox(LinearPolygonMesh<N, D, T, I> const & mesh) noexcept
    -> AxisAlignedBox<D, T>
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
    box = um2::boundingBox(box, mesh.getFace(i).boundingBox());
  }    
  return box;    
}

// -----------------------------------------------------------------------------
// faceContaining(Point)
// -----------------------------------------------------------------------------
template <Size P, Size N, std::floating_point T, std::signed_integral I>
PURE constexpr auto
faceContaining(PlanarPolygonMesh<P, N, T, I> const & mesh, Point2<T> const & p)
    noexcept -> Size
{
  for (Size i = 0; i < numFaces(mesh); ++i) {
    if (mesh.getFace(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1;
}

// -----------------------------------------------------------------------------
// toFaceVertexMesh(MeshFile)
// -----------------------------------------------------------------------------

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
// Return true if the MeshType and P, N are compatible.
template <Size P, Size N>
constexpr auto validateMeshFileType(MeshType const /*type*/) -> bool
{
  return false;
}

template <>
constexpr
auto validateMeshFileType<1, 3>(MeshType const type) -> bool
{
  return type == MeshType::Tri;
}

template <>
constexpr
auto validateMeshFileType<1, 4>(MeshType const type) -> bool
{
  return type == MeshType::Quad;
}

template <>
constexpr
auto validateMeshFileType<2, 6>(MeshType const type) -> bool
{
  return type == MeshType::QuadraticTri;
}

template <>
constexpr
auto validateMeshFileType<2, 8>(MeshType const type) -> bool
{
  return type == MeshType::QuadraticQuad;
}

template <Size P, Size N>
constexpr
auto getMeshType() -> MeshType
{
  return MeshType::None;
}

template <>
constexpr
auto getMeshType<1, 3>() -> MeshType
{
  return MeshType::Tri;
}

template <>
constexpr
auto getMeshType<1, 4>() -> MeshType
{
  return MeshType::Quad;
}

template <>
constexpr
auto getMeshType<2, 6>() -> MeshType
{
  return MeshType::QuadraticTri;
}

template <>
constexpr
auto getMeshType<2, 8>() -> MeshType
{
  return MeshType::QuadraticQuad;
}

#pragma GCC diagnostic pop

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void toFaceVertexMesh(
    MeshFile<T, I> const & file, 
    FaceVertexMesh<P, N, D, T, I> & mesh) noexcept
{
  assert(!file.vertices.empty());
  assert(!file.element_conn.empty());
  auto const num_vertices = static_cast<Size>(file.vertices.size());    
  auto const num_faces = static_cast<Size>(file.numCells());    
  auto const conn_size = static_cast<Size>(file.element_conn.size());    
  auto const verts_per_face = verticesPerCell(file.type);
  if (!validateMeshFileType<P, N>(file.type)) {
    Log::error(    
        "Attempted to construct a FaceVertexMesh from a mesh file with an incompatible mesh type");    
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
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(), mesh.vf_offsets.begin() + 1);
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
  for (Size i = 0; i < num_vertices - 1; ++i) {
    if (isApprox(vertices_copy[i], vertices_copy[i + 1])) {
      Log::warn("Vertex " + std::to_string(i) + " and " + std::to_string(i + 1) +
                " are effectively equivalent");
    }
  }

  // Check that the vertices are in counter-clockwise order.
  // If the area of the face is negative, then the vertices are in clockwise
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
      if (!mesh.getFace(i).isConvex()) {
        Log::warn("Face " + std::to_string(i) + " is not convex");
      }
    }
  }

  // Overlap check
#endif
}


template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void toMeshFile(
    FaceVertexMesh<P, N, D, T, I> const & mesh,
    MeshFile<T, I> & file) noexcept
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
  // NOLINTBEGIN(bugprone-misplaced-widening-cast)
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

#include <um2/mesh/face_vertex_mesh/getFace.inl>
