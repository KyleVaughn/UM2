#pragma once

#include <um2/geometry/quadratic_quadrilateral.hpp>
#include <um2/geometry/quadratic_triangle.hpp>
#include <um2/geometry/quadrilateral.hpp>
#include <um2/geometry/triangle.hpp>
#include <um2/mesh/polytope_soup.hpp>

//=============================================================================
// FACE-VERTEX MESH
//=============================================================================
// A 2D volumetric mesh composed of polygons of polynomial order P.
// Each polygon (face) is composed of N vertices. Each vertex is a 2-dimensional
// point. See polytope.hpp for more information on the polygon data structure.
//  - P = 1, N = 3: Triangular mesh
//  - P = 1, N = 4: Quadrilateral mesh
//  - P = 2, N = 6: Quadratic triangular mesh
//  - P = 2, N = 8: Quadratic quadrilateral mesh
//
// Let Int be the signed integer type used to index vertices and faces.
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
// In short:
// - _v[i] is the i-th vertex
// - _fv[i] is the Vec of vertex ids of the i-th face
// - _vf_offsets[i] is the index of the start of the i-th vertex's face list in _vf.
//   _vf_offsets[i + 1] - _vf_offsets[i] is the number of faces to which vertex i belongs
//
// ASSUMPTIONS:
// - Faces are oriented counter-clockwise
// - Manifold mesh

namespace um2
{

template <Int P, Int N>
class FaceVertexMesh
{

public:
  using FaceConn = Vec<N, Int>;
  using EdgeConn = Vec<P + 1, Int>;
  using Face = Polygon<P, N, 2, Float>;
  using Edge = typename Polygon<P, N, 2, Float>::Edge;
  using Vertex = typename Polygon<P, N, 2, Float>::Vertex;

private:
  bool _is_morton_ordered = false;
  bool _has_vf = false;
  Vector<Vertex> _v;       // vertices
  Vector<FaceConn> _fv;    // face-vertex connectivity
  Vector<Int> _vf_offsets; // index into _vf
  Vector<Int> _vf;         // vertex-face connectivity

public:
  //===========================================================================
  // Constructors
  //===========================================================================

  constexpr FaceVertexMesh() noexcept = default;

  HOSTDEV
  constexpr FaceVertexMesh(Vector<Vertex> const & v,
                           Vector<FaceConn> const & fv) noexcept;

  explicit FaceVertexMesh(PolytopeSoup const & soup);

  //===========================================================================
  // Member access
  //===========================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() noexcept -> Vector<Vertex> &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() const noexcept -> Vector<Vertex> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  faceVertexConn() noexcept -> Vector<FaceConn> &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  faceVertexConn() const noexcept -> Vector<FaceConn> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertexFaceOffsets() const noexcept -> Vector<Int> const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertexFaceConn() const noexcept -> Vector<Int> const &;

  //===========================================================================
  // Capacity
  //===========================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numVertices() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numFaces() const noexcept -> Int;

  //===========================================================================
  // Getters
  //===========================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getVertex(Int i) const noexcept -> Vertex;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdge(Int iface, Int iedge) const noexcept -> Edge;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getEdgeConn(Int iface, Int iedge) const noexcept -> EdgeConn;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFace(Int i) const noexcept -> Face;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getFaceConn(Int i) const noexcept -> FaceConn const &;

  //===========================================================================
  // Modifiers
  //===========================================================================

  constexpr void
  addVertex(Vertex const & v) noexcept;

  constexpr void
  addFace(FaceConn const & conn) noexcept;

  constexpr void
  flipFace(Int i) noexcept;

  void
  mortonSort() noexcept;

  void
  mortonSortFaces() noexcept;

  void
  mortonSortVertices() noexcept;

  void
  populateVF() noexcept;

  //===========================================================================
  // Methods
  //===========================================================================

  PURE [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox2F;

  PURE [[nodiscard]] constexpr auto
  faceContaining(Point2F p) const noexcept -> Int;

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator PolytopeSoup() const noexcept;

  void
  validate();

  // Intersect the mesh with a ray.
  // Store the parametric ray coordinates of the intersections in coords.
  // Return the number of intersections.
  auto
  intersect(Ray2F ray, Float * coords) const noexcept -> Int;

  // Intersect the mesh with a ray.
  // Store the parametric ray coordinates of the intersections in coords.
  // Store the starting index of the i-th face's coordinates in offsets.
  //   coords[offsets[i]] is the first coordinate of the i-th face's intersection.
  //   offsets[i + 1] - offsets[i] is the number of intersections with the i-th face.
  // Store the IDs of the intersected faces in faces.
  // Return the (number of intersections, number of faces).
  auto
  intersect(Ray2F ray, Float * coords, Int * RESTRICT offsets,
            Int * RESTRICT faces) const noexcept -> Vec2I;
};

//==============================================================================
// Aliases
//==============================================================================

template <Int P, Int N>
using FVM = FaceVertexMesh<P, N>;

// Polynomial order
template <Int N>
using LinearFVM = FVM<1, N>;
template <Int N>
using QuadraticFVM = FVM<2, N>;

// Number of vertices per face
using TriFVM = LinearFVM<3>;
using QuadFVM = LinearFVM<4>;
using Tri6FVM = QuadraticFVM<6>;
using Quad8FVM = QuadraticFVM<8>;

//==============================================================================
// Free functions
//==============================================================================

// Due to the variable sizes of the number of intersections with each face,
// sorting in-place without allocating memory is difficult. Instead, we will
// use additional buffers to store sorted coords, sorted offsets, and faces,
// as well as a permutation array.
void
sortRayMeshIntersections(Float const * RESTRICT coords,  // size >= total_hits
                         Int const * RESTRICT offsets,   // size >= num_faces + 1
                         Int const * RESTRICT faces,     // size >= num_faces
                         Float * RESTRICT sorted_coords, // size >= total_hits
                         Int * RESTRICT sorted_offsets,  // size >= num_faces + 1
                         Int * RESTRICT sorted_faces,    // size >= num_faces
                         Int * RESTRICT perm,            // size >= num_faces
                         Vec2I hits_faces                // (total_hits, num_faces)
);

//==============================================================================
// Constructors
//==============================================================================

template <Int P, Int N>
constexpr FaceVertexMesh<P, N>::FaceVertexMesh(Vector<Vertex> const & v,
                                               Vector<FaceConn> const & fv) noexcept
    : _v(v),
      _fv(fv)
{
  validate();
}

//==============================================================================
// Member access
//==============================================================================

template <Int P, Int N>
HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::vertices() noexcept -> Vector<Vertex> &
{
  return _v;
}

template <Int P, Int N>
HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::vertices() const noexcept -> Vector<Vertex> const &
{
  return _v;
}

template <Int P, Int N>
PURE HOSTDEV constexpr auto
FaceVertexMesh<P, N>::faceVertexConn() noexcept -> Vector<FaceConn> &
{
  return _fv;
}

template <Int P, Int N>
PURE HOSTDEV constexpr auto
FaceVertexMesh<P, N>::faceVertexConn() const noexcept -> Vector<FaceConn> const &
{
  return _fv;
}

template <Int P, Int N>
PURE HOSTDEV constexpr auto
FaceVertexMesh<P, N>::vertexFaceOffsets() const noexcept -> Vector<Int> const &
{
  return _vf_offsets;
}

template <Int P, Int N>
PURE HOSTDEV constexpr auto
FaceVertexMesh<P, N>::vertexFaceConn() const noexcept -> Vector<Int> const &
{
  return _vf;
}

//==============================================================================
// Capacity
//==============================================================================

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::numVertices() const noexcept -> Int
{
  return _v.size();
}

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::numFaces() const noexcept -> Int
{
  return _fv.size();
}

//==============================================================================
// Getters
//==============================================================================

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::getVertex(Int i) const noexcept -> Vertex
{
  return _v[i];
}

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::getEdge(Int iface, Int iedge) const noexcept -> Edge
{
  static_assert(P == 1 || P == 2);
  ASSERT_ASSUME(0 <= iface);
  ASSERT(iface < numFaces());
  ASSERT_ASSUME(0 <= iedge);
  Int constexpr num_edges = polygonNumEdges<P, N>();
  ASSERT_ASSUME(iedge < num_edges);
  auto const & conn = _fv[iface];
  if constexpr (P == 1) {
    // equivalent to getEdge(LinearPolygon<N, 2> const & p, Int iedge)
    return (iedge < num_edges - 1) ? Edge(_v[conn[iedge]], _v[conn[iedge + 1]])
                                   : Edge(_v[conn[N - 1]], _v[conn[0]]);
  } else if constexpr (P == 2) {
    // equivalent to getEdge(QuadraticPolygon<N, 2> const & p, Int iedge)
    return (iedge < num_edges - 1)
               ? Edge(_v[conn[iedge]], _v[conn[iedge + 1]], _v[conn[iedge + num_edges]])
               : Edge(_v[conn[num_edges - 1]], _v[conn[0]], _v[conn[N - 1]]);
  } else {
    __builtin_unreachable();
  }
}

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::getEdgeConn(Int iface, Int iedge) const noexcept -> EdgeConn
{
  static_assert(P == 1 || P == 2);
  ASSERT_ASSUME(0 <= iface);
  ASSERT(iface < numFaces());
  ASSERT_ASSUME(0 <= iedge);
  Int constexpr num_edges = polygonNumEdges<P, N>();
  ASSERT_ASSUME(iedge < num_edges);
  auto const & conn = _fv[iface];
  if constexpr (P == 1) {
    // equivalent to getEdge(LinearPolygon<N, 2> const & p, Int iedge)
    return (iedge < num_edges - 1) ? EdgeConn(conn[iedge], conn[iedge + 1])
                                   : EdgeConn(conn[N - 1], conn[0]);
  } else if constexpr (P == 2) {
    // equivalent to getEdge(QuadraticPolygon<N, 2> const & p, Int iedge)
    return (iedge < num_edges - 1)
               ? EdgeConn(conn[iedge], conn[iedge + 1], conn[iedge + num_edges])
               : EdgeConn(conn[num_edges - 1], conn[0], conn[N - 1]);
  } else {
    __builtin_unreachable();
  }
}

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::getFace(Int i) const noexcept -> Face
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < numFaces());
  return {_fv[i], _v.data()};
}

template <Int P, Int N>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N>::getFaceConn(Int i) const noexcept -> FaceConn const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < numFaces());
  return _fv[i];
}

//==============================================================================
// Modifiers
//==============================================================================

template <Int P, Int N>
constexpr void
FaceVertexMesh<P, N>::addVertex(Vertex const & v) noexcept
{
  _has_vf = false; // Invalidate vertex-face connectivity
  _v.emplace_back(v);
}

template <Int P, Int N>
constexpr void
FaceVertexMesh<P, N>::addFace(FaceConn const & conn) noexcept
{
  _has_vf = false; // Invalidate vertex-face connectivity
  _fv.emplace_back(conn);
}

template <Int P, Int N>
constexpr void
FaceVertexMesh<P, N>::flipFace(Int i) noexcept
{
  _has_vf = false; // Invalidate vertex-face connectivity
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
// Methods
//==============================================================================

template <Int N>
PURE [[nodiscard]] constexpr auto
boundingBox(LinearFVM<N> const & mesh) noexcept -> AxisAlignedBox2F
{
  auto const & vertices = mesh.vertices();
  return um2::boundingBox(vertices.cbegin(), vertices.cend());
}

template <Int N>
PURE [[nodiscard]] constexpr auto
boundingBox(QuadraticFVM<N> const & mesh) noexcept -> AxisAlignedBox2F
{
  auto box = mesh.getFace(0).boundingBox();
  for (Int i = 1; i < mesh.numFaces(); ++i) {
    box += mesh.getFace(i).boundingBox();
  }
  return box;
}

template <Int P, Int N>
PURE constexpr auto
FaceVertexMesh<P, N>::boundingBox() const noexcept -> AxisAlignedBox2F
{
  return um2::boundingBox(*this);
}

template <Int P, Int N>
PURE constexpr auto
FaceVertexMesh<P, N>::faceContaining(Point2F const p) const noexcept -> Int
{
  for (Int i = 0; i < numFaces(); ++i) {
    if (getFace(i).contains(p)) {
      return i;
    }
  }
  return -1;
}

template <Int P, Int N>
auto
FaceVertexMesh<P, N>::intersect(Ray2F const ray,
                                Float * const coords) const noexcept -> Int
{
  Int hits = 0;
  for (Int i = 0; i < numFaces(); ++i) {
    hits += getFace(i).intersect(ray, coords + hits);
  }
  return hits;
}

template <Int P, Int N>
auto
FaceVertexMesh<P, N>::intersect(Ray2F const ray, Float * coords, Int * RESTRICT offsets,
                                Int * RESTRICT faces) const noexcept -> Vec2I
{
  *offsets++ = 0;
  Int total_hits = 0;
  Int num_faces = 0;
  for (Int i = 0; i < numFaces(); ++i) {
    Int const hits = getFace(i).intersect(ray, coords + total_hits);
    if (hits > 0) {
      total_hits += hits;
      ++num_faces;
      *offsets++ = total_hits;
      *faces++ = i;
    }
  }
  return {total_hits, num_faces};
}

} // namespace um2
