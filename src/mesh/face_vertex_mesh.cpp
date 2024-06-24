#include <um2/config.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/polytope_soup.hpp>

#include <um2/common/logger.hpp>
#include <um2/common/permutation.hpp>
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/geometry/point.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/vector.hpp>
#include <um2/math/vec.hpp>

#include <algorithm> // sort
#include <numeric> // inclusive_scan

#include <iostream> // std::cerr

namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

// Helper function to get VTK element type from P and N.

namespace {
template <Int P, Int N>
constexpr auto
getVTKElemType() -> VTKElemType
{
  if constexpr (P == 1 && N == 3) {
    return VTKElemType::Triangle;
  } else if constexpr (P == 1 && N == 4) {
    return VTKElemType::Quad;
  } else if constexpr (P == 2 && N == 6) {
    return VTKElemType::QuadraticTriangle;
  } else if constexpr (P == 2 && N == 8) {
    return VTKElemType::QuadraticQuad;
  }
  return VTKElemType::Invalid;
}
} // namespace

template <Int P, Int N>
FaceVertexMesh<P, N>::FaceVertexMesh(PolytopeSoup const & soup)
{
  auto const num_vertices = soup.numVertices();
  auto const num_faces = soup.numElements();
  ASSERT(num_vertices > 0);
  ASSERT(num_faces > 0);
  auto const elem_types = soup.getElemTypes();
  if (elem_types.size() != 1) { 
    logger::error("Attempted to construct a FaceVertexMesh from a non-homogeneous PolytopeSoup");
  }
  if (elem_types[0] != getVTKElemType<P, N>()) {
    logger::error("Attempted to construct a FaceVertexMesh from a PolytopeSoup with an incompatible element type");
  }

  // -- Vertices --
  // Ensure each of the vertices has approximately the same z
  _v.resize(num_vertices);
  Float const z = soup.getVertex(0)[2];
  for (Int i = 0; i < num_vertices; ++i) {
    auto const & p = soup.getVertex(i);
    _v[i][0] = p[0];
    _v[i][1] = p[1];
    if (um2::abs(p[2] - z) > epsDistance<Float>()) {
      logger::warn("Constructing a FaceVertexMesh from a PolytopeSoup with non-planar vertices");
      break;
    }
  }

  // -- Face/Vertex connectivity --
  _fv.resize(num_faces);
  auto const & soup_conn = soup.elementConnectivity();
  for (Int i = 0; i < num_faces; ++i) {
    for (Int j = 0; j < N; ++j) {
      _fv[i][j] = soup_conn[i * N + j];
    }
  }
  validate();
}

//==============================================================================
// Modifiers 
//==============================================================================

template <Int P, Int N>
void
FaceVertexMesh<P, N>::mortonSort() noexcept
{
  LOG_DEBUG("Sorting vertices and faces using morton encoding");
  // If the mesh had vertex-face connectivity, need to invalidate it, then
  // recompute it.
  bool const had_vf = _has_vf;
  mortonSortVertices();
  mortonSortFaces();
  _is_morton_ordered = true;
  if (had_vf) {
    populateVF();
  }
}

template <Int P, Int N>
void
FaceVertexMesh<P, N>::mortonSortFaces() noexcept
{
  // Invalidate vertex-face connectivity.
  _has_vf = false;

  // Sort the centroid of each face using the morton encoding.
  Int const num_faces = numFaces();
  Vector<Point2F> centroids(num_faces);
  for (Int i = 0; i < num_faces; ++i) {
    centroids[i] = getFace(i).centroid();
  }
  // We need to scale the centroids to the unit cube before we can apply
  // the morton encoding. Therefore we need to find the bounding box of
  // all faces.
  auto const aabb = boundingBox();
  Vec2F const inv_scale = 1 / aabb.extents();

  // Get the permutation vector which sorts the centroids according to the morton
  // encoding.
  Vector<Int> perm(num_faces);
  mortonSortPermutation(centroids.begin(), centroids.end(), perm.begin(), inv_scale);

  // Sort the faces according to the permutation vector.
  applyPermutation(_fv.begin(), _fv.end(), perm.cbegin());
}

template <Int P, Int N>
void
FaceVertexMesh<P, N>::mortonSortVertices() noexcept
{
  // Invalidate vertex-face connectivity
  _has_vf = false;

  // We need to scale the vertices to the unit cube before we can apply
  // the morton encoding.
  auto const aabb = boundingBox();
  Vec2F const inv_scale = 1 / aabb.extents();

  // Get the permutation vector which sorts the vertices according to the morton
  // encoding. We also need the inverse permutation to ensure that the face-vertex
  // connectivity is maintained.
  Int const num_verts = numVertices();
  Vector<Int> perm(num_verts);
  Vector<Int> inv_perm(num_verts);
  mortonSortPermutation(_v.cbegin(), _v.cend(), perm.begin(), inv_scale);
  invertPermutation(perm.cbegin(), perm.cend(), inv_perm.begin());

  // Sort the vertices according to the permutation vector.
  applyPermutation(_v.begin(), _v.end(), perm.cbegin());

  // Map the old vertex indices to the new vertex indices.
  for (auto & face : _fv) {
    for (auto & vert_id : face) {
      vert_id = inv_perm[vert_id];
    }
  }
}

template <Int P, Int N>
void
FaceVertexMesh<P, N>::populateVF() noexcept
{
  // Make no assumption about _vf_offsets and _vf being empty.
  Int const num_vertices = numVertices();
  Int const num_faces = numFaces();

  // Count the occurrences of each vertex in the face-vertex list.
  Vector<Int> vert_counts(num_vertices, 0);
  for (Int i = 0; i < num_faces; ++i) {
    for (Int j = 0; j < N; ++j) {
      // fv[i][j] is the j-th vertex index of the i-th face.
      ++vert_counts[_fv[i][j]];
    }
  }

  // Compute the offsets
  _vf_offsets.resize(num_vertices + 1);
  _vf_offsets[0] = 0;
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(), _vf_offsets.begin() + 1);

  // Populate the vertex-face list
  // For each face, for each vertex, add the add the face to the vertex-face list.
  _vf.resize(_vf_offsets[num_vertices]);
  // Copy vf_offsets to vert_offsets
  Vector<Int> vert_offsets = _vf_offsets;
  for (Int i = 0; i < num_faces; ++i) {
    auto const & face = _fv[i];
    for (Int j = 0; j < N; ++j) {
      auto const vert = face[j];
      // vert_offsets[vert] is the index where the next face will be added.
      _vf[vert_offsets[vert]] = i;
      ++vert_offsets[vert];
    }
  }
  // Note: Since the faces were added in order, the vertex-face list is sorted at
  // each vertex.
  _has_vf = true;
}

//==============================================================================
// Methods 
//==============================================================================

template <Int P, Int N>
FaceVertexMesh<P, N>::operator PolytopeSoup() const noexcept
{
  PolytopeSoup soup;

  // Vertices
  soup.reserveMoreVertices(numVertices());
  for (Int i = 0; i < numVertices(); ++i) {
    soup.addVertex(_v[i][0], _v[i][1]);
  }

  // Faces
  VTKElemType const elem_type = getVTKElemType<P, N>();
  Vector<Int> conn(N);
  soup.reserveMoreElements(elem_type, numFaces());
  for (Int i = 0; i < numFaces(); ++i) {
    for (Int j = 0; j < N; ++j) {
      conn[j] = _fv[i][j];
    }
    soup.addElement(elem_type, conn);
  }
  return soup;
}

namespace {

template <Int P, Int N>
void
checkCCWFaces(FaceVertexMesh<P, N> & mesh)
{
  // Check that the vertices are in counter-clockwise order.
  Int const num_faces = mesh.numFaces();
  bool faces_flipped = false;
  for (Int i = 0; i < num_faces; ++i) {
    if (!mesh.getFace(i).isCCW()) {
      mesh.flipFace(i);
      faces_flipped = true;
    }
  }
  if (faces_flipped) {
    logger::warn("Some faces were flipped to ensure counter-clockwise order");
  }
}

template <Int P, Int N>
void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
checkManifoldWatertight(FaceVertexMesh<P, N> const & mesh)
{
  // Ensure that the mesh doesn't have any holes or self-intersections.
  // Algorithm:
  //  1. For each face, get the edges as represented by the vertex indices.
  //  2. Sort the edges by the smallest vertex index (for quadratic edges, use the
  //      linear vertex index).
  //  3. Count the number of times each edge/orientation occurs.
  //    - If edge (i, j) occurs more than once, then the mesh has overlapping faces.
  //    - If edge (i, j) and edge (j, i) occue exactly once, then the edge is an
  //        interior edge.
  //    - If edge (i, j) occurs exactly once, then the edge is a boundary edge.
  //  4. Get the boundary edges.
  //    - If the mesh does not have any holes, then the number of boundary edges should
  //      by able to form a single closed loop.
  //    - Determine the number of closed loops.

  using EdgeConn = typename FaceVertexMesh<P, N>::EdgeConn;

  Int const num_faces = mesh.numFaces();
  Int constexpr edges_per_face = polygonNumEdges<P, N>();
  Int const total_num_edges = num_faces * edges_per_face;

  // 1
  //---------------------------------------------------------------------------
  // (Edge conn, orientation) pairs. +1 if ordered, -1 if reversed.
  Vector<Pair<EdgeConn, Int>> edge_conns(total_num_edges);
  for (Int iface = 0; iface < num_faces; ++iface) {
    for (Int iedge = 0; iedge < edges_per_face; ++iedge) {
      auto & edge_conn = edge_conns[iface * edges_per_face + iedge];
      edge_conn.first = mesh.getEdgeConn(iface, iedge);
      ASSERT(edge_conn.first[0] != edge_conn.first[1]);
      // Ensure that the first vertex index is less than the second.
      if (edge_conn.first[0] > edge_conn.first[1]) {
        um2::swap(edge_conn.first[0], edge_conn.first[1]);
        edge_conn.second = -1;
      } else {
        edge_conn.second = 1;
      }
    }
  }

  // 2
  //---------------------------------------------------------------------------
  std::sort(edge_conns.begin(), edge_conns.end());
  // edge_conns is now sorted by edge, then orientation.

  // 3
  //---------------------------------------------------------------------------
  // We expect num_unique_edges >= total_num_edges / 2, approaching equality
  // as the ratio of boundary edges / interior edges approaches 0.
  Vector<EdgeConn> unique_edges;
  Vector<Vec2I> edge_counts;
  unique_edges.reserve(total_num_edges / 2);
  edge_counts.reserve(total_num_edges / 2);

  // push back the first edge and count
  if (edge_conns[0].second == 1) {
    edge_counts.emplace_back(0, 1);
    unique_edges.emplace_back(edge_conns[0].first);
  } else {
    edge_counts.emplace_back(1, 0);
    // We want the natural ordering of the edge
    unique_edges.emplace_back(edge_conns[0].first);
    auto & edge = unique_edges.back();
    um2::swap(edge[0], edge[1]);
  }

  for (Int i = 1; i < total_num_edges; ++i) {
    auto const & prev_edge_conn = edge_conns[i - 1];
    auto const & edge_conn = edge_conns[i];
    // If this edge is the same as the previous edge, increment the count.
    if (prev_edge_conn.first == edge_conn.first) {
      auto & edge_counts_back = edge_counts.back();
      // Increment the orientation count.
      if (edge_conn.second == 1) {
        ++edge_counts_back[1];
      } else {
        ++edge_counts_back[0];
      }
    } else {
      // Otherwise, emplace back the edge.
      if (edge_conn.second == 1) {
        edge_counts.emplace_back(0, 1);
        unique_edges.emplace_back(edge_conn.first);
      } else {
        edge_counts.emplace_back(1, 0);
        unique_edges.emplace_back(edge_conn.first);
        auto & edge = unique_edges.back();
        um2::swap(edge[0], edge[1]);
      }
    }
  }

  // 4
  //---------------------------------------------------------------------------

  //- If edge (i, j) occurs more than once, then the mesh has overlapping faces.
  //- If edge (i, j) and edge (j, i) occue exactly once, then the edge is an
  //    interior edge.
  //- If edge (i, j) occurs exactly once, then the edge is a boundary edge.
  Vector<EdgeConn> boundary_edges;
  for (Int i = 0; i < edge_counts.size(); ++i) {
    auto const & edge_count = edge_counts[i];
    Int const sum = edge_count[0] + edge_count[1];
    if (sum > 2) {
      logger::error("Mesh has overlapping faces");
      return;
    }
    if (sum == 1) {
      // Ensure that the first and second vertex indices are unique.
      // Otherwise, the mesh has a hole.
      auto const & edge = unique_edges[i];
      for (auto const & bedge : boundary_edges) {
        if (edge[0] == bedge[0] || edge[1] == bedge[1]) {
          logger::error("Mesh has a hole on its boundary");
          return;
        }
      }
      boundary_edges.emplace_back(edge);
    }
  }

  // We expect the number of boundary edges to be relatively small compared to the
  // total number of edges. Hence, we can afford to use a simple O(n^2) algorithm.
  Int ctr = 0; // The number of edges in the boundary loop.
  Int const start_idx = boundary_edges[0][0];
  auto edge = boundary_edges[0];
  while (true) {
    // Find the edge that has the same start index as the current edge's end index.
    bool edge_found = false;
    for (auto const & bedge : boundary_edges) {
      if (edge[1] == bedge[0]) {
        edge = bedge;
        ++ctr;
        edge_found = true;
        break;
      }
    }
    // If the edge does not exist, there is a boundary edge missing.
    if (!edge_found) {
      logger::error("Mesh has a hanging boundary edge");
      return;
    }
    // If we're back at the start, we have a closed loop.
    if (edge[0] == start_idx) {
      break;
    }
  }
  // If the number of edges in the boundary loop is not equal to the number of
  // boundary edges, then the mesh has multiple boundary loops.
  if (ctr != boundary_edges.size()) {
    logger::error("Mesh has a hole in its interior");
    return;
  }
} // checkManifoldWatertight

template <Int N>
void
checkSelfIntersections(FaceVertexMesh<2, N> const & mesh)
{
  Int const num_faces = mesh.numFaces();
  Point2F buffer[2 * N];
  for (Int iface = 0; iface < num_faces; ++iface) {
    if (mesh.getFace(iface).hasSelfIntersection(buffer)) {
      PolytopeSoup const soup = mesh;
      soup.write("self_intersecting_mesh.xdmf");
      std::cerr << "Intersection at (" << buffer[0][0] << ", " << buffer[0][1] << ") or ("
                << buffer[1][0] << ", " << buffer[1][1] << ")" << std::endl;
      logger::error("Mesh has self-intersecting face at index: ", iface, ". Mesh written to self_intersecting_mesh.xdmf");
      return;
    }
  }
}

} // namespace

// Check for:
// - Counter-clockwise faces (warn and fix)
// - Manifoldness/watertight (error)
// - Self-intersections (error, Quadratic elements only)
template <Int P, Int N>
void
FaceVertexMesh<P, N>::validate()
{
  // Check that the vertices are in counter-clockwise order.
  checkCCWFaces(*this);

  // Check that the mesh is manifold and watertight.
  checkManifoldWatertight(*this);

  if constexpr (P == 2) {
    checkSelfIntersections(*this);
  }
}

//==============================================================================
// Free functions
//==============================================================================

void
sortRayMeshIntersections(
    Float const * RESTRICT coords,    // size >= total_hits                 
    Int const * RESTRICT offsets,     // size >= num_faces + 1 
    Int const * RESTRICT faces,       // size >= num_faces 
    Float * RESTRICT sorted_coords,   // size >= total_hits    
    Int * RESTRICT sorted_offsets,    // size >= num_faces + 1          
    Int * RESTRICT sorted_faces,      // size >= num_faces            
    Int * RESTRICT perm,              // size >= num_faces        
    Vec2I hits_faces                  // (total_hits, num_faces)
)
{
  // Sort based on the intersection coordinates.
  // Coordinate comparison:
  // - Can't use smallest coordinate only, since floating point error might cause
  //   faces to be out of order.
  // - Can't use average of coordinates, since a symmetric face which bounds
  //   another face can have the same average coordinate.
  //
  // Therefore, we use the average of the two smallest coordinates. Note, in some cases
  // due to floating point error, a face may have only 1 intersection coordinate.
  // However, we expect an even number of intersection coordinates.

  // Compute the average of the first two coordinates.
  // We only need these values to create the permutation array so
  // we will store them in the sorted_coords buffer.
  Int const num_faces = hits_faces[1];
  for (Int iface = 0; iface < num_faces; ++iface) {
    Int const offset = offsets[iface];
    Int const next_offset = offsets[iface + 1];
    Float r0 = coords[offset]; 
    Float r1 = coords[offset];
    for (Int i = offset + 1; i < next_offset; ++i) {
      Float const r = coords[i];
      if (r < r0) {
        r1 = r0;
        r0 = r;
      } else if (r < r1) {
        r1 = r;
      }
    }
    sorted_coords[iface] = (r0 + r1) / 2; 
  }

  // Obtain the permutation vector.
  sortPermutation(sorted_coords, sorted_coords + num_faces, perm);

  // Apply the permutation vector to the faces and coords.
  sorted_offsets[0] = 0;
  for (Int iface = 0; iface < num_faces; ++iface) {
    Int const ind = perm[iface];
    Int const size = offsets[ind + 1] - offsets[ind];
    sorted_offsets[iface + 1] = sorted_offsets[iface] + size;
    sorted_faces[iface] = faces[ind];
    for (Int i = 0; i < size; ++i) {
      sorted_coords[sorted_offsets[iface] + i] = coords[offsets[ind] + i];
    }
    // sort the coordinates for this face
    std::sort(sorted_coords + sorted_offsets[iface], sorted_coords + sorted_offsets[iface + 1]);
  }
}
//==============================================================================
// Explicit instantiations
//==============================================================================

template class FaceVertexMesh<1, 3>; // TriFVM
template class FaceVertexMesh<1, 4>; // QuadFVM
template class FaceVertexMesh<2, 6>; // Tri6FVM
template class FaceVertexMesh<2, 8>; // Quad8FVM

} // namespace um2
