#include <um2/geometry/morton_sort_points.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>

namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

// Return true if the MeshType and P, N are compatible.
template <Size P, Size N>
static constexpr auto
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

template <Size P, Size N>
static constexpr auto
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
  ASSERT(false);
  return VTKElemType::None;
}

template <Size P, Size N>
FaceVertexMesh<P, N>::FaceVertexMesh(PolytopeSoup const & soup)
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
  _v.resize(num_vertices);
#if UM2_ENABLE_ASSERTS
  F constexpr eps = eps_distance;
  F const z = soup.getVertex(0)[2];
  for (Size i = 1; i < num_vertices; ++i) {
    ASSERT(um2::abs(soup.getVertex(i)[2] - z) < eps);
  }
#endif
  for (Size i = 0; i < num_vertices; ++i) {
    auto const & p = soup.getVertex(i);
    _v[i][0] = p[0];
    _v[i][1] = p[1];
  }

  // -- Face/Vertex connectivity --
  Vector<I> conn(N);
  VTKElemType elem_type = VTKElemType::None;
  _fv.resize(num_faces);
  for (Size i = 0; i < num_faces; ++i) {
    soup.getElement(i, elem_type, conn);
    ASSERT(elem_type == (getVTKElemType<P, N>()));
    for (Size j = 0; j < N; ++j) {
      _fv[i][j] = conn[j];
    }
  }
  validate();
}

//==============================================================================
// addVertex
//==============================================================================

template <Size P, Size N>
void
FaceVertexMesh<P, N>::addVertex(Point2 const & v) noexcept
{
  _v.push_back(v);
}

//==============================================================================
// addFace
//==============================================================================

template <Size P, Size N>
void
FaceVertexMesh<P, N>::addFace(FaceConn const & conn) noexcept
{
  _fv.push_back(conn);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size P, Size N>
PURE [[nodiscard]] auto
FaceVertexMesh<P, N>::boundingBox() const noexcept -> AxisAlignedBox2
{
  if constexpr (P == 1) {
    return um2::boundingBox(_v);
  } else if constexpr (P == 2) {
    auto box = getFace(0).boundingBox();
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

 template <Size P, Size N>
PURE [[nodiscard]] auto
FaceVertexMesh<P, N>::faceContaining(Point2 const & p) const noexcept
    -> Size
{
  for (Size i = 0; i < numFaces(); ++i) {
    if (getFace(i).contains(p)) {
      return i;
    }
  }
  return -1;
}

//==============================================================================
// flipFace
//==============================================================================

template <Size P, Size N>
void
FaceVertexMesh<P, N>::flipFace(Size i) noexcept
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
// mortonSort
//==============================================================================

template <Size P, Size N>
void
FaceVertexMesh<P, N>::mortonSort()
{
  LOG_DEBUG("Sorting vertices and faces using morton encoding");
  mortonSortVertices();
  mortonSortFaces();
  _is_morton_sorted = true;
}

//==============================================================================
// mortonSortFaces
//==============================================================================

template <Size P, Size N>
void
FaceVertexMesh<P, N>::mortonSortFaces()
{
  // We will sort the centroid of each face using the morton encoding.
  Size const num_faces = numFaces();
  Vector<Point2> centroids(num_faces);
  for (Size i = 0; i < num_faces; ++i) {
    centroids[i] = getFace(i).centroid();
  }
  // We need to scale the centroids to the unit cube before we can apply
  // the morton encoding. Therefore we need to find the bounding box of
  // all faces.
  auto aabb = um2::boundingBox(_v);
  for (Size i = 0; i < num_faces; ++i) {
    aabb += getFace(i).boundingBox();
  }
  Point2 inv_scale = aabb.maxima() - aabb.minima();
  inv_scale[0] = static_cast<F>(1) / inv_scale[0];
  inv_scale[1] = static_cast<F>(1) / inv_scale[1];

  for (auto & c : centroids) {
    c *= inv_scale;
  }

  // Create a vector of Morton codes for the centroids.
  Vector<MortonCode> morton_codes(num_faces, 0);
  for (Size i = 0; i < num_faces; ++i) {
    morton_codes[i] = mortonEncode(centroids[i]);
  }

  // Create a vector of indices into the centroids vector.
  Vector<Size> perm(num_faces);

  // Sort the indices as to create a permutation vector.
  // perm[new_index] = old_index
  sortPermutation(morton_codes.cbegin(), morton_codes.cend(), perm.begin());

  // Sort the faces according to the permutation vector.
  applyPermutation(_fv, perm);

  // Invalidate vertex-face connectivity.
  _has_vf = false;
}

//==============================================================================
// mortonSortVertices
//==============================================================================

 template <Size P, Size N>
void
FaceVertexMesh<P, N>::mortonSortVertices()
{
  // We need to scale the vertices to the unit cube before we can apply
  // the morton encoding.
  auto const aabb = um2::boundingBox(_v);
  Point2 inv_scale = aabb.maxima() - aabb.minima();
  inv_scale[0] = static_cast<F>(1) / inv_scale[0];
  inv_scale[1] = static_cast<F>(1) / inv_scale[1];
  Size const num_verts = numVertices();
  Vector<Point2> scaled_verts(num_verts);
  for (Size i = 0; i < num_verts; ++i) {
    scaled_verts[i] = _v[i];
    scaled_verts[i] *= inv_scale;
  }

  // Create a vector of Morton codes for the vertices.
  Vector<MortonCode> morton_codes(num_verts, 0);
  for (Size i = 0; i < num_verts; ++i) {
    morton_codes[i] = mortonEncode(scaled_verts[i]);
  }

  // Create a vector of indices into the vertices vector.
  Vector<Size> perm(num_verts);

  // Sort the indices as to create a permutation vector.
  // perm[new_index] = old_index
  sortPermutation(morton_codes.cbegin(), morton_codes.cend(), perm.begin());
  ASSERT(!um2::is_sorted(morton_codes.cbegin(), morton_codes.cend()));

  // We also want the inverse of the permutation vector.
  // inv_perm[old_index] = new_index
  // inv_perm[perm[new_index]] = new_index
  Vector<Size> inv_perm(num_verts);
  invertPermutation(perm, inv_perm);

  // Sort the vertices according to the permutation vector.
  applyPermutation(_v, perm);

  // Map the old vertex indices to the new vertex indices.
  for (auto & face : _fv) {
    for (auto & vert_id : face) {
      vert_id = inv_perm[vert_id];
    }
  }

  // Invalidate vertex-face connectivity
  _has_vf = false;
}

//==============================================================================
// populateVF
//==============================================================================

 template <Size P, Size N>
void
FaceVertexMesh<P, N>::populateVF() noexcept
{
  // Make no assumption about _vf_offsets and _vf being empty.
  Size const num_vertices = numVertices();
  Size const num_faces = numFaces();

  // -- Vertex/Face connectivity --
  Vector<I> vert_counts(num_vertices, 0);
  for (Size i = 0; i < num_faces; ++i) {
    for (Size j = 0; j < N; ++j) {
      ++vert_counts[_fv[i][j]];
    }
  }
  _vf_offsets.resize(num_vertices + 1);
  _vf_offsets[0] = 0;
  std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(), _vf_offsets.begin() + 1);
  _vf.resize(_vf_offsets[num_vertices]);
  // Copy vf_offsets to vert_offsets
  Vector<I> vert_offsets = _vf_offsets;
  for (Size i = 0; i < num_faces; ++i) {
    auto const & face = _fv[i];
    for (Size j = 0; j < N; ++j) {
      auto const vert = face[j];
      _vf[vert_offsets[vert]] = i;
      ++vert_offsets[vert];
    }
  }
  _has_vf = true;
}

////// //==============================================================================
////// // toPolytopeSoup
////// //==============================================================================
//////
//////  template <Size P, Size N>
////// void
////// toPolytopeSoup(FaceVertexMesh<P, N> const & mesh,
//////                PolytopeSoup & soup) noexcept
////// {
//////   // Vertices
//////   if constexpr (D == 3) {
//////     for (Size i = 0; i < mesh.numVertices(); ++i) {
//////       soup.addVertex(mesh.vertices[i]);
//////     }
//////   } else {
//////     for (Size i = 0; i < mesh.numVertices(); ++i) {
//////       auto const & p = mesh.vertices[i];
//////       soup.addVertex(p[0], p[1]);
//////     }
//////   }
//////
//////   // Faces
//////   auto const nfaces = mesh.numFaces();
//////   VTKElemType const elem_type = getVTKElemType<P, N>();
//////   Vector<I> conn(N);
//////   for (Size i = 0; i < nfaces; ++i) {
//////     for (Size j = 0; j < N; ++j) {
//////       conn[j] = mesh.fv[i][j];
//////     }
//////     soup.addElement(elem_type, conn);
//////   }
////// }
//////
//////  template <Size P, Size N>
////// void
////// FaceVertexMesh<P, N>::toPolytopeSoup(PolytopeSoup & soup) const noexcept
////// {
//////   um2::toPolytopeSoup(*this, soup);
////// }
//////
//==============================================================================
// intersect
//==============================================================================

template <Size N>
void
intersect(Ray2 const & ray, LinearFVM<N> const & mesh,
          Vector<F> & intersections) noexcept
{
  Size constexpr edges_per_face = LinearFVM<N>::Face::numEdges();
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < edges_per_face; ++j) {
      auto const edge = face.getEdge(j);
      F const r = intersect(ray, edge);
      if (r < inf_distance) {
        intersections.push_back(r);
      }
    }
  }
  std::sort(intersections.begin(), intersections.end());
}

template <Size N>
void
intersect(Ray2 const & ray, QuadraticFVM<N> const & mesh,
          Vector<F> & intersections) noexcept
{
  Size constexpr edges_per_face = QuadraticFVM<N>::Face::numEdges();
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < edges_per_face; ++j) {
      auto const edge = face.getEdge(j);
      auto const r = intersect(ray, edge);
      if (r[0] < inf_distance) {
        intersections.push_back(r[0]);
      }
      if (r[1] < inf_distance) {
        intersections.push_back(r[1]);
      }
    }
  }
  std::sort(intersections.begin(), intersections.end());
}


////// Input:
////// intersections: A buffer of size n
////// n: The max size of the buffer
//////
////// Output:
////// intersections: The buffer is filled with n intersections
////// n: The number of intersections
////template <Size N, std::floating_point T, std::signed_integral I>
////void
////intersectFixedBuffer(Ray2<F> const & ray, PlanarLinearPolygonMesh<N, T, I> const & mesh,
////                     F * const intersections, Size * const n) noexcept
////{
////  Size nintersect = 0;
////#if UM2_ENABLE_ASSERTS
////  Size const n0 = *n;
////#endif
////  Size constexpr edges_per_face = PlanarQuadraticPolygonMesh<N, T, I>::Face::numEdges();
////  for (Size i = 0; i < mesh.numFaces(); ++i) {
////    auto const face = mesh.getFace(i);
////    for (Size j = 0; j < edges_per_face; ++j) {
////      auto const edge = face.getEdge(j);
////      auto const r = intersect(ray, edge);
////      if (r < inf_distance) {
////        ASSERT(nintersect < n0)
////        intersections[nintersect++] = r;
////      }
////    }
////  }
////  *n = nintersect;
////  std::sort(intersections, intersections + nintersect);
////}
////
////// Input:
////// intersections: A buffer of size n
////// n: The max size of the buffer
//////
////// Output:
////// intersections: The buffer is filled with n intersections
////// n: The number of intersections
////template <Size N, std::floating_point T, std::signed_integral I>
////void
////intersectFixedBuffer(Ray2<F> const & ray,
////                     PlanarQuadraticPolygonMesh<N, T, I> const & mesh,
////                     F * const intersections, Size * const n) noexcept
////{
////  Size nintersect = 0;
////#if UM2_ENABLE_ASSERTS
////  Size const n0 = *n;
////#endif
////  Size constexpr edges_per_face = PlanarQuadraticPolygonMesh<N, T, I>::Face::numEdges();
////  //  for (Size i = 0; i < mesh.numFaces(); ++i) {
////  //    auto const face = mesh.getFace(i);
////  //    for (Size j = 0; j < edges_per_face; ++j) {
////  //      auto const edge = face.getEdge(j);
////  //      auto const r = intersect(ray, edge);
////  //      if (r[0] < inf_distance) {
////  //        ASSERT(nintersect < n0)
////  //        intersections[nintersect++] = r[0];
////  //      }
////  //      if (r[1] < inf_distance) {
////  //        ASSERT(nintersect < n0)
////  //        intersections[nintersect++] = r[1];
////  //      }
////  //    }
////  //  }
////  for (Size i = 0; i < mesh.numFaces(); ++i) {
////    for (Size j = 0; j < edges_per_face; ++j) {
////      auto const edge = mesh.getEdge(i, j);
////      auto const r = intersect(ray, edge);
////      if (r[0] < inf_distance) {
////        ASSERT(nintersect < n0)
////        intersections[nintersect++] = r[0];
////      }
////      if (r[1] < inf_distance) {
////        ASSERT(nintersect < n0)
////        intersections[nintersect++] = r[1];
////      }
////    }
////  }
////
////  *n = nintersect;
////  std::sort(intersections, intersections + nintersect);
////}

 template <Size P, Size N>
void
FaceVertexMesh<P, N>::intersect(Ray2 const & ray,
                                         Vector<F> & intersections) const noexcept
{
  um2::intersect(ray, *this, intersections);
}

//==============================================================================
// validate
//==============================================================================

// Check for:
// - Repeated vertices (warn)
// - Counter-clockwise faces (warn and fix)
// - Convexity (warn, quad mesh only)
template <Size P, Size N>
void
FaceVertexMesh<P, N>::validate()
{
#if UM2_ENABLE_ASSERTS
  // Check for repeated vertices.
  // This is not technically an error, but it is a sign that the mesh may
  // cause problems for some algorithms. Hence, we warn the user.
  auto const bbox = boundingBox();
  auto const minima = bbox.minima();
  auto const maxima = bbox.maxima();
  Vec2<F> normalization;
  normalization[0] = static_cast<F>(1) / (maxima[0] - minima[0]);
  normalization[1] = static_cast<F>(1) / (maxima[1] - minima[1]);
  Vector<Point2> vertices_copy = _v;
  // Transform the points to be in the unit cube
  for (auto & v : vertices_copy) {
    v -= minima;
    v *= normalization;
  }
  um2::mortonSort(vertices_copy.begin(), vertices_copy.end());
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

//==============================================================================
// Explicit instantiations
//==============================================================================

template class FaceVertexMesh<1, 3>; // TriFVM
template class FaceVertexMesh<1, 4>; // QuadFVM
template class FaceVertexMesh<2, 6>; // Tri6FVM
template class FaceVertexMesh<2, 8>; // Quad8FVM

template void intersect(Ray2 const & ray, LinearFVM<3> const & mesh,
                        Vector<F> & intersections) noexcept;
template void intersect(Ray2 const & ray, LinearFVM<4> const & mesh,
                        Vector<F> & intersections) noexcept;
template void intersect(Ray2 const & ray, QuadraticFVM<6> const & mesh,
                        Vector<F> & intersections) noexcept;
template void intersect(Ray2 const & ray, QuadraticFVM<8> const & mesh,
                        Vector<F> & intersections) noexcept;

} // namespace um2
