#pragma once

#include <um2/mesh/polytope_soup.hpp>

//=============================================================================
// FACE-VERTEX MESH
//=============================================================================
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

namespace um2
{

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
class FaceVertexMesh {

  public:
  using FaceConn = Vec<N, I>;
  using Face = Polygon<P, N, D, T>;
  using Edge = typename Polygon<P, N, D, T>::Edge;

  private:
  bool _is_morton_sorted = false;
  bool _has_vf = false;
  Vector<Point<D, T>> _v; // vertices
  Vector<FaceConn> _fv;   // face-vertex connectivity
  Vector<I> _vf_offsets;  // A prefix sum of the number of faces to which each
                          // vertex belongs. size = num_vertices + 1
  Vector<I> _vf;          // vertex-face connectivity

  public:

  //===========================================================================
  // Constructors
  //===========================================================================

  constexpr FaceVertexMesh() noexcept = default;

  constexpr FaceVertexMesh(Vector<Point<D, T>> const & v,
                           Vector<FaceConn> const & fv) noexcept;

  explicit FaceVertexMesh(PolytopeSoup<T, I> const & soup);

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numVertices() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numFaces() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getVertex(Size i) const noexcept -> Point<D, T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFace(Size i) const noexcept -> Face;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getVFOffsets() const noexcept -> Vector<I> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getVF() const noexcept -> Vector<I> const &;

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

  void
  populateVF() noexcept;

  void
  validate();
  //
  //  void
  //  toPolytopeSoup(PolytopeSoup<T, I> & soup) const noexcept;
  //
  //  //  //  void
  //  //  //  getFaceAreas(Vector<T> & areas) const noexcept;
  //  //  //
  //  //  //  void
  //  //  //  getUniqueEdges(Vector<Edge> & edges) const noexcept;
  //  //  //
  void
  intersect(Ray<D, T> const & ray, Vector<T> & intersections) const noexcept
    requires(D == 2);
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
// Constructors
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
constexpr
FaceVertexMesh<P, N, D, T, I>::FaceVertexMesh(Vector<Point<D, T>> const & v,
                                              Vector<FaceConn> const & fv) noexcept
  : _v(v), _fv(fv)
{
}

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
  return false;
}

// template <Size P, Size N>
// constexpr auto
// getMeshType() -> MeshType
//{
//   if constexpr (P == 1 && N == 3) {
//     return MeshType::Tri;
//   } else if constexpr (P == 1 && N == 4) {
//     return MeshType::Quad;
//   } else if constexpr (P == 2 && N == 6) {
//     return MeshType::QuadraticTri;
//   } else if constexpr (P == 2 && N == 8) {
//     return MeshType::QuadraticQuad;
//   }
//   ASSERT(false);
//   return MeshType::None;
// }
//
// template <Size P, Size N>
// constexpr auto
// getVTKElemType() -> VTKElemType
// {
//   if constexpr (P == 1 && N == 3) {
//     return VTKElemType::Triangle;
//   } else if constexpr (P == 1 && N == 4) {
//     return VTKElemType::Quad;
//   } else if constexpr (P == 2 && N == 6) {
//     return VTKElemType::QuadraticTriangle;
//   } else if constexpr (P == 2 && N == 8) {
//     return VTKElemType::QuadraticQuad;
//   }
//   ASSERT(false);
//   return VTKElemType::None;
// }

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
FaceVertexMesh<P, N, D, T, I>::FaceVertexMesh(PolytopeSoup<T, I> const & soup)
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
#if UM2_ENABLE_ASSERTS
    T const eps = eps_distance<T>;
    T const z = soup.getVertex(0)[2];
    for (Size i = 1; i < num_vertices; ++i) {
      ASSERT(um2::abs(soup.getVertex(i)[2] - z) < eps);
    }
#endif
    _v.resize(num_vertices);
    for (Size i = 0; i < num_vertices; ++i) {
      auto const & p = soup.getVertex(i);
      _v[i][0] = p[0];
      _v[i][1] = p[1];
    }
  } else {
    for (Size i = 0; i < num_vertices; ++i) {
      _v[i] = soup.getVertex(i);
    }
  }

  // -- Face/Vertex connectivity --
  Vector<I> conn(N);
  VTKElemType elem_type = VTKElemType::None;
  _fv.resize(num_faces);
  for (Size i = 0; i < num_faces; ++i) {
    soup.getElement(i, elem_type, conn);
    for (Size j = 0; j < N; ++j) {
      _fv[i][j] = conn[j];
    }
  }
  validate();
}

//==============================================================================
// Accessors
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numVertices() const noexcept -> Size
{
  return _v.size();
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numFaces() const noexcept -> Size
{
  return _fv.size();
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::getVertex(Size i) const noexcept -> Point<D, T>
{
  return _v[i];
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::getFace(Size i) const noexcept -> Face
{
  if constexpr (P == 1 && N == 3) {
    return Triangle<D, T>(_v[static_cast<Size>(_fv[i][0])],
                          _v[static_cast<Size>(_fv[i][1])],
                          _v[static_cast<Size>(_fv[i][2])]);
  } else if constexpr (P == 1 && N == 4) {
    return Quadrilateral<D, T>(_v[static_cast<Size>(_fv[i][0])],
                               _v[static_cast<Size>(_fv[i][1])],
                               _v[static_cast<Size>(_fv[i][2])],
                               _v[static_cast<Size>(_fv[i][3])]);
  } else if constexpr (P == 2 && N == 6) {
    return QuadraticTriangle<D, T>(_v[static_cast<Size>(_fv[i][0])],
                                   _v[static_cast<Size>(_fv[i][1])],
                                   _v[static_cast<Size>(_fv[i][2])],
                                   _v[static_cast<Size>(_fv[i][3])],
                                   _v[static_cast<Size>(_fv[i][4])],
                                   _v[static_cast<Size>(_fv[i][5])]);
  } else if constexpr (P == 2 && N == 8) {
    return QuadraticQuadrilateral<D, T>(_v[static_cast<Size>(_fv[i][0])],
                                        _v[static_cast<Size>(_fv[i][1])],
                                        _v[static_cast<Size>(_fv[i][2])],
                                        _v[static_cast<Size>(_fv[i][3])],
                                        _v[static_cast<Size>(_fv[i][4])],
                                        _v[static_cast<Size>(_fv[i][5])],
                                        _v[static_cast<Size>(_fv[i][6])],
                                        _v[static_cast<Size>(_fv[i][7])]);
  } else {
    __builtin_unreachable();
  }
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
FaceVertexMesh<P, N, D, T, I>::getVFOffsets() const noexcept -> Vector<I> const &
{
  return _vf_offsets;
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
FaceVertexMesh<P, N, D, T, I>::getVF() const noexcept -> Vector<I> const &
{
  return _vf;
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  if constexpr (P == 1) {
    return um2::boundingBox(_v);
  } else if constexpr (P == 2) {
    AxisAlignedBox<D, T> box = getFace(0).boundingBox();
    for (Size i = 1; i < numFaces(); ++i) {
      box += getFace(i).boundingBox();
    }
    return box;
  } else {
    __builtin_unreachable();
  }
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
  for (Size i = 0; i < numFaces(); ++i) {
    if (getFace(i).contains(p)) {
      return i;
    }
  }
  ASSERT(false);
  return -1;
}

//==============================================================================
// flipFace
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::flipFace(Size i) noexcept
{
  if constexpr (P == 1 && N == 3) {
    um2::swap(_fv[i][1], _fv[i][2]);
  } else if constexpr (P == 1 && N == 4) {
    um2::swap(_fv[i][1], _fv[i][3]);
  } else if constexpr (P == 2 && N == 6) {
    um2::swap(_fv[i][1], _fv[i][2]);
    um2::swap(_fv[i][3], _fv[i][5]);
  } else if constexpr (P == 2 && N == 8) {
    um2::swap(_fv[i][1], _fv[i][3]);
    um2::swap(_fv[i][4], _fv[i][7]);
  }
}

//==============================================================================
// populateVF
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::populateVF() noexcept
{
  Size const num_vertices = numVertices();
  Size const num_faces = numFaces();

  // -- Vertex/Face connectivity --
  Vector<I> vert_counts(num_vertices, 0);
  for (Size i = 0; i < num_faces; ++i) {
    for (Size j = 0; j < N; ++j) {
      ++vert_counts[static_cast<Size>(_fv[i][j])];
    }
  }
  _vf_offsets.resize(num_vertices + 1);
  _vf_offsets[0] = 0;
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(), _vf_offsets.begin() + 1);
  _vf.resize(static_cast<Size>(_vf_offsets[num_vertices]));
  // Copy vf_offsets to vert_offsets
  Vector<I> vert_offsets = _vf_offsets;
  for (Size i = 0; i < num_faces; ++i) {
    auto const & face = _fv[i];
    for (Size j = 0; j < N; ++j) {
      auto const vert = static_cast<Size>(face[j]);
      _vf[static_cast<Size>(vert_offsets[vert])] = static_cast<I>(i);
      ++vert_offsets[vert];
    }
  }
}

// //==============================================================================
// // toPolytopeSoup
// //==============================================================================
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// toPolytopeSoup(FaceVertexMesh<P, N, D, T, I> const & mesh,
//                PolytopeSoup<T, I> & soup) noexcept
// {
//   // Vertices
//   if constexpr (D == 3) {
//     for (Size i = 0; i < mesh.numVertices(); ++i) {
//       soup.addVertex(mesh.vertices[i]);
//     }
//   } else {
//     for (Size i = 0; i < mesh.numVertices(); ++i) {
//       auto const & p = mesh.vertices[i];
//       soup.addVertex(p[0], p[1]);
//     }
//   }
//
//   // Faces
//   auto const nfaces = mesh.numFaces();
//   VTKElemType const elem_type = getVTKElemType<P, N>();
//   Vector<I> conn(N);
//   for (Size i = 0; i < nfaces; ++i) {
//     for (Size j = 0; j < N; ++j) {
//       conn[j] = mesh.fv[i][j];
//     }
//     soup.addElement(elem_type, conn);
//   }
// }
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// FaceVertexMesh<P, N, D, T, I>::toPolytopeSoup(PolytopeSoup<T, I> & soup) const noexcept
// {
//   um2::toPolytopeSoup(*this, soup);
// }
//
//==============================================================================
// intersect
//==============================================================================

template <Size N, std::floating_point T, std::signed_integral I>
void
intersect(Ray2<T> const & ray, PlanarLinearPolygonMesh<N, T, I> const & mesh,
          Vector<T> & intersections) noexcept
{
  Size constexpr edges_per_face = PlanarLinearPolygonMesh<N, T, I>::Face::numEdges();
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < edges_per_face; ++j) {
      auto const edge = face.getEdge(j);
      T const r = intersect(ray, edge);
      if (r < inf_distance<T>) {
        intersections.push_back(r);
      }
    }
  }
  std::sort(intersections.begin(), intersections.end());
}

template <Size N, std::floating_point T, std::signed_integral I>
void
intersect(Ray2<T> const & ray, PlanarQuadraticPolygonMesh<N, T, I> const & mesh,
          Vector<T> & intersections) noexcept
{
  Size constexpr edges_per_face = PlanarQuadraticPolygonMesh<N, T, I>::Face::numEdges();
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < edges_per_face; ++j) {
      auto const edge = face.getEdge(j);
      auto const r = intersect(ray, edge);
      if (r[0] < inf_distance<T>) {
        intersections.push_back(r[0]);
      }
      if (r[1] < inf_distance<T>) {
        intersections.push_back(r[1]);
      }
    }
  }
  std::sort(intersections.begin(), intersections.end());
}

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::intersect(Ray<D, T> const & ray, Vector<T> & intersections) const noexcept
  requires(D == 2)
{
  um2::intersect(ray, *this, intersections);
}

//==============================================================================
// validate
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::validate()
{
#if UM2_ENABLE_ASSERTS
  // Check for repeated vertices.
  // This is not technically an error, but it is a sign that the mesh may
  // cause problems for some algorithms. Hence, we warn the user.
  auto const bbox = boundingBox();
  auto const minima = bbox.minima();
  auto const maxima = bbox.maxima();
  Vec<D, T> normalization;
  for (Size i = 0; i < D; ++i) {
    normalization[i] = static_cast<T>(1) / (maxima[i] - minima[i]);
  }
  Vector<Point<D, T>> vertices_copy = _v;
  // Transform the points to be in the unit cube
  for (auto & v : vertices_copy) {
    v -= minima;
    v *= normalization;
  }
  if constexpr (std::same_as<T, float>) {
    mortonSort<uint32_t>(vertices_copy.begin(), vertices_copy.end());
  } else {
    mortonSort<uint64_t>(vertices_copy.begin(), vertices_copy.end());
  }
  // Revert the scaling
  for (auto & v : vertices_copy) {
    v /= normalization;
  }
  Size const num_vertices = numVertices();
  for (Size i = 0; i < num_vertices - 1; ++i) {
    if (isApprox(vertices_copy[i], vertices_copy[i + 1])) {
      Log::warn("Vertex " + toString(i) + " and " + toString(i + 1) +
                " are effectively equivalent");
    }
  }
#endif

  // Check that the vertices are in counter-clockwise order.
  Size const num_faces = numFaces();
  for (Size i = 0; i < num_faces; ++i) {
    if (!getFace(i).isCCW()) {
      Log::warn("Face " + toString(i) + " has vertices in clockwise order. Reordering");
      flipFace(i);
    }
  }

  // Convexity check
  if constexpr (N == 4) {
    for (Size i = 0; i < num_faces; ++i) {
      if (!isApproxConvex(getFace(i))) {
        Log::warn("Face " + toString(i) + " is not convex");
      }
    }
  }
}

} // namespace um2
