#pragma once

#include <um2/mesh/FaceVertexMesh.hpp>
#include <um2/mesh/RegularPartition.hpp>

namespace um2
{

//=============================================================================
// BINNED FACE-VERTEX MESH
//=============================================================================
//
// A 2D volumetric or 3D surface mesh composed of polygons of polynomial order P.
// Each polygon (face) is composed of N vertices. Each vertex is a D-dimensional
// point of floating point type T.
//
// The faces of the mesh are sorted into a regular grid of bins based upon the
// upper right corner of the bounding box of the face. The bin size is determined
// by the bounding box of the mesh and the size of the largest face bounding box.
// In direction x, the number of bins is:
//  num_bins_x = floor((max_x - min_x) / max_box_length)
//
// face_ids is a Vector<I> of size numFaces(mesh) that holds the ids of the faces
// in each bin. partition.children holds the offsets into face_ids for each bin.
//
// To get the faces in bin (i, j, k), use:
// auto const flat_index = partition.getFlatIndex(i, j, k);
// auto const & index_start = partition.children[flat_index];
// auto const & index_end = partition.children[flat_index + 1];
// for (Size index = index_start; index < index_end; ++index) {
//  auto const face_id = face_ids[index];
//  auto const & face = mesh.getFace(face_id);
//  // Do something with face.
//  ...
// }
//
template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
struct BinnedFaceVertexMesh {

  FaceVertexMesh<P, N, D, T, I> mesh;
  RegularPartition<D, T, I> partition;
  Vector<I> face_ids;

  //===========================================================================
  // Constructors
  //===========================================================================

  constexpr BinnedFaceVertexMesh() noexcept = default;

  explicit BinnedFaceVertexMesh(FaceVertexMesh<P, N, D, T, I> const & mesh_in) noexcept
      : mesh(mesh_in)
  {
    // Get the bounding box of the mesh.
    auto const box = boundingBox(mesh);

    // Determine the maximum edge length in the mesh.
    Size const nfaces = numFaces(mesh);
    T max_side_length = 0;
    for (Size i = 0; i < nfaces; ++i) {
      auto const face = mesh.getFace(i);
      auto const face_bb = face.boundingBox();
      auto const delta = face_bb.maxima - face_bb.minima;
      for (Size j = 0; j < D; ++j) {
        max_side_length = um2::max(max_side_length, delta[j]);
      }
    }
    // std::cerr << "max_side_length = " << max_side_length << std::endl;
    //     Log::info("max_edge_length = " + std::to_string(max_edge_length));

    // Determine the number of bins and the bin size in each dimension.
    Vec<D, Size> num_bins;
    Vec<D, T> bin_size;
    Size total_bins = 1;
    for (Size d = 0; d < D; ++d) {
      T const delta = box.maxima[d] - box.minima[d];
      num_bins[d] = static_cast<Size>(um2::floor(delta / max_side_length));
      total_bins *= num_bins[d];
      assert(num_bins[d] > 0);
      bin_size[d] = delta / static_cast<T>(num_bins[d]);
    }
    // std::cerr << "num_bins = (" << num_bins[0] << ", " << num_bins[1] << ")" <<
    // std::endl;

    // Initialize the partition's grid.
    partition.grid.minima = box.minima;
    partition.grid.spacing = bin_size;
    partition.grid.num_cells = num_bins;
    // Children stores the offsets into face_ids for each bin, so we need
    // 1 more element than the total number of bins.
    // We count the number of faces in each bin and store the offsets in
    // partition.children.
    // To obtain the offsets perform an exclusive scan on partition.children.
    partition.children.resize(total_bins + 1);
    partition.children[0] = 0; // Just to be safe.
    for (Size i = 0; i < nfaces; ++i) {
      auto const upper_right_point = mesh.getFace(i).boundingBox().maxima;
      auto const index = partition.getCellIndexContaining(upper_right_point);
      Size const flat_index = partition.getFlatIndex(index);
      ++partition.children[flat_index + 1];
    }
    // Perform an exclusive scan on partition.children to get the offsets.
    for (Size i = 1; i < partition.children.size(); ++i) {
      partition.children[i] += partition.children[i - 1];
    }
    assert(std::is_sorted(partition.children.begin(), partition.children.end()));
    // Allocate space for the face ids.
    face_ids = um2::move(Vector<I>(nfaces, -1));
    // Assign the face ids to the bins.
    for (Size i = 0; i < nfaces; ++i) {
      auto const upper_right_point = mesh.getFace(i).boundingBox().maxima;
      auto const index = partition.getCellIndexContaining(upper_right_point);
      Size const offset_index = partition.getFlatIndex(index);
      auto offset = static_cast<Size>(partition.children[offset_index]);
      // Find the first empty slot in the bin.
      while (face_ids[offset] != -1) {
        ++offset;
      }
      face_ids[offset] = static_cast<I>(i);
    }
    assert(std::all_of(face_ids.begin(), face_ids.end(),
                       [](auto const & id) { return id != -1; }));
  }

  //===========================================================================
  // Methods
  //===========================================================================
  // PURE [[nodiscard]] constexpr auto
  // faceContaining(Point<D, T> const & p) const noexcept -> Size
  //{
  //  return um2::faceContaining(*this, p);
  //}
};

//==============================================================================
// Aliases
//==============================================================================

// Polynomial order
template <Size N, Size D, std::floating_point T, std::signed_integral I>
using BinnedLinearPolygonMesh = BinnedFaceVertexMesh<1, N, D, T, I>;
template <Size N, Size D, std::floating_point T, std::signed_integral I>
using BinnedQuadraticPolygonMesh = BinnedFaceVertexMesh<2, N, D, T, I>;

// Number of vertices per face
template <Size D, std::floating_point T, std::signed_integral I>
using BinnedTriMesh = BinnedLinearPolygonMesh<3, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using BinnedQuadMesh = BinnedLinearPolygonMesh<4, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using BinnedQuadraticTriMesh = BinnedQuadraticPolygonMesh<6, D, T, I>;
template <Size D, std::floating_point T, std::signed_integral I>
using BinnedQuadraticQuadMesh = BinnedQuadraticPolygonMesh<8, D, T, I>;

// 2D
template <Size P, Size N, std::floating_point T, std::signed_integral I>
using BinnedPlanarPolygonMesh = BinnedFaceVertexMesh<P, N, 2, T, I>;

//==============================================================================
// Methods
//==============================================================================
// For all FaceVertexMesh, we define:
//   numVertices
////   numFaces
////   getFace
////   boundingBox
////   faceContaining(Point)
////   toFaceVertexMesh(MeshFile)
////   toMeshFile(FaceVertexMesh)

////==============================================================================
//// numVertices
////==============================================================================
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// PURE HOSTDEV constexpr auto
// numVertices(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
//{
//  return mesh.vertices.size();
//}
//
////==============================================================================
//// numFaces
////==============================================================================
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// PURE HOSTDEV constexpr auto
// numFaces(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
//{
//  return mesh.fv.size();
//}
//
////==============================================================================
//// boundingBox
////==============================================================================
//
// template <Size N, Size D, std::floating_point T, std::signed_integral I>
// PURE constexpr auto
// boundingBox(LinearPolygonMesh<N, D, T, I> const & mesh) noexcept -> AxisAlignedBox<D,
// T>
//{
//  return boundingBox(mesh.vertices);
//}
//
////==============================================================================
//// boundingBox
////==============================================================================
//
// template <Size N, Size D, std::floating_point T, std::signed_integral I>
// PURE constexpr auto
// boundingBox(QuadraticPolygonMesh<N, D, T, I> const & mesh) noexcept
//    -> AxisAlignedBox<D, T>
//{
//  AxisAlignedBox<D, T> box = mesh.getFace(0).boundingBox();
//  for (Size i = 1; i < numFaces(mesh); ++i) {
//    box += mesh.getFace(i).boundingBox();
//  }
//  return box;
//}
//
//==============================================================================
// faceContaining(Point)
//==============================================================================

template <Size P, Size N, std::floating_point T, std::signed_integral I>
PURE constexpr auto
faceContaining(BinnedPlanarPolygonMesh<P, N, T, I> const & bmesh,
               Point2<T> const & p) noexcept -> Size
{
  // auto const flat_index = partition.getFlatIndex(i, j, k);
  // auto const & index_start = partition.children[flat_index];
  // auto const & index_end = partition.children[flat_index + 1];
  // for (Size index = index_start; index < index_end; ++index) {
  //  auto const face_id = face_ids[index];
  //  auto const & face = mesh.getFace(face_id);
  //  // Do something with face.
  //  ...
  // }

  // std::cerr << "p = (" << p[0] << ", " << p[1] << ")\n";
  auto const index = bmesh.partition.getCellIndexContaining(p);
  // std::cerr << "index = (" << index[0] << ", " << index[1] << ")\n";
  {
    Size const offset_index = bmesh.partition.getFlatIndex(index);
    auto const offset_start = static_cast<Size>(bmesh.partition.children[offset_index]);
    auto const offset_end = static_cast<Size>(bmesh.partition.children[offset_index + 1]);
    assert(offset_start <= offset_end);
    for (Size offset = offset_start; offset < offset_end; ++offset) {
      auto const i = static_cast<Size>(bmesh.face_ids[offset]);
      // std::cerr << "i = " << i << "\n";
      if (bmesh.mesh.getFace(i).contains(p)) {
        return i;
      }
    }
  }

  bool const is_last_on_x = bmesh.partition.numXCells() == index[0] + 1;
  if (!is_last_on_x) {
    auto index_right = index;
    ++index_right[0];
    // std::cerr << "index_right = (" << index_right[0] << ", " << index_right[1] <<
    // ")\n";
    Size const offset_index = bmesh.partition.getFlatIndex(index_right);
    auto const offset_start = static_cast<Size>(bmesh.partition.children[offset_index]);
    auto const offset_end = static_cast<Size>(bmesh.partition.children[offset_index + 1]);
    assert(offset_start <= offset_end);
    for (Size offset = offset_start; offset < offset_end; ++offset) {
      auto const i = static_cast<Size>(bmesh.face_ids[offset]);
      // std::cerr << "i = " << i << "\n";
      if (bmesh.mesh.getFace(i).contains(p)) {
        return i;
      }
    }
  }
  bool const is_last_on_y = bmesh.partition.numYCells() == index[1] + 1;
  if (!is_last_on_y) {
    auto index_top = index;
    ++index_top[1];
    // std::cerr << "index_top = (" << index_top[0] << ", " << index_top[1] << ")\n";
    Size const offset_index = bmesh.partition.getFlatIndex(index_top);
    auto const offset_start = static_cast<Size>(bmesh.partition.children[offset_index]);
    auto const offset_end = static_cast<Size>(bmesh.partition.children[offset_index + 1]);
    assert(offset_start <= offset_end);
    for (Size offset = offset_start; offset < offset_end; ++offset) {
      auto const i = static_cast<Size>(bmesh.face_ids[offset]);
      // std::cerr << "i = " << i << "\n";
      if (bmesh.mesh.getFace(i).contains(p)) {
        return i;
      }
    }
  }
  if (!is_last_on_x && !is_last_on_y) {
    auto index_top_right = index;
    ++index_top_right[0];
    ++index_top_right[1];
    // std::cerr << "index_top_right = (" << index_top_right[0] << ", " <<
    // index_top_right[1]
    //           << ")\n";
    Size const offset_index = bmesh.partition.getFlatIndex(index_top_right);
    auto const offset_start = static_cast<Size>(bmesh.partition.children[offset_index]);
    auto const offset_end = static_cast<Size>(bmesh.partition.children[offset_index + 1]);
    assert(offset_start <= offset_end);
    for (Size offset = offset_start; offset < offset_end; ++offset) {
      auto const i = static_cast<Size>(bmesh.face_ids[offset]);
      // std::cerr << "i = " << i << "\n";
      if (bmesh.mesh.getFace(i).contains(p)) {
        return i;
      }
    }
  }
  // std::cerr << "Actual face = " << bmesh.mesh.faceContaining(p) << "\n";
  assert(false);
  return -1;
}

////==============================================================================
//// toFaceVertexMesh(MeshFile)
////==============================================================================
//
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wunused-function"
//// Return true if the MeshType and P, N are compatible.
// template <Size P, Size N>
// constexpr auto
// validateMeshFileType(MeshType const /*type*/) -> bool
//{
//   return false;
// }
//
// template <>
// constexpr auto
// validateMeshFileType<1, 3>(MeshType const type) -> bool
//{
//   return type == MeshType::Tri;
// }
//
// template <>
// constexpr auto
// validateMeshFileType<1, 4>(MeshType const type) -> bool
//{
//   return type == MeshType::Quad;
// }
//
// template <>
// constexpr auto
// validateMeshFileType<2, 6>(MeshType const type) -> bool
//{
//   return type == MeshType::QuadraticTri;
// }
//
// template <>
// constexpr auto
// validateMeshFileType<2, 8>(MeshType const type) -> bool
//{
//   return type == MeshType::QuadraticQuad;
// }
//
// template <Size P, Size N>
// constexpr auto
// getMeshType() -> MeshType
//{
//   return MeshType::None;
// }
//
// template <>
// constexpr auto
// getMeshType<1, 3>() -> MeshType
//{
//   return MeshType::Tri;
// }
//
// template <>
// constexpr auto
// getMeshType<1, 4>() -> MeshType
//{
//   return MeshType::Quad;
// }
//
// template <>
// constexpr auto
// getMeshType<2, 6>() -> MeshType
//{
//   return MeshType::QuadraticTri;
// }
//
// template <>
// constexpr auto
// getMeshType<2, 8>() -> MeshType
//{
//   return MeshType::QuadraticQuad;
// }
//
// #pragma GCC diagnostic pop
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// void
//// NOLINTNEXTLINE(readability-function-cognitive-complexity)
// toFaceVertexMesh(MeshFile<T, I> const & file,
//                  FaceVertexMesh<P, N, D, T, I> & mesh) noexcept
//{
//   assert(!file.vertices.empty());
//   assert(!file.element_conn.empty());
//   auto const num_vertices = static_cast<Size>(file.vertices.size());
//   auto const num_faces = static_cast<Size>(file.numCells());
//   auto const conn_size = static_cast<Size>(file.element_conn.size());
//   if (!validateMeshFileType<P, N>(file.type)) {
//     Log::error("Attempted to construct a FaceVertexMesh from a mesh file with an "
//                "incompatible mesh type");
//   }
//   assert(conn_size == num_faces * verticesPerCell(file.type));
//
//   // -- Vertices --
//   // Ensure each of the vertices has approximately the same z
//   if constexpr (D == 2) {
// #ifndef NDEBUG
//     T const eps = eps_distance<T>;
//     T const z = file.vertices[0][2];
//     for (auto const & v : file.vertices) {
//       assert(std::abs(v[2] - z) < eps);
//     }
// #endif
//     mesh.vertices.resize(num_vertices);
//     for (Size i = 0; i < num_vertices; ++i) {
//       mesh.vertices[i][0] = file.vertices[static_cast<size_t>(i)][0];
//       mesh.vertices[i][1] = file.vertices[static_cast<size_t>(i)][1];
//     }
//   } else {
//     mesh.vertices = file.vertices;
//   }
//
//   // -- Face/Vertex connectivity --
//   mesh.fv.resize(num_faces);
//   for (Size i = 0; i < num_faces; ++i) {
//     for (Size j = 0; j < N; ++j) {
//       auto const idx = i * N + j;
//       mesh.fv[i][j] = file.element_conn[static_cast<size_t>(idx)];
//     }
//   }
//
//   // -- Vertex/Face connectivity --
//   Vector<I> vert_counts(num_vertices, 0);
//   for (size_t i = 0; i < static_cast<size_t>(conn_size); ++i) {
//     ++vert_counts[static_cast<Size>(file.element_conn[i])];
//   }
//   mesh.vf_offsets.resize(num_vertices + 1);
//   mesh.vf_offsets[0] = 0;
//   std::inclusive_scan(vert_counts.cbegin(), vert_counts.cend(),
//                       mesh.vf_offsets.begin() + 1);
//   vert_counts.clear();
//   mesh.vf.resize(static_cast<Size>(mesh.vf_offsets[num_vertices]));
//   // Copy vf_offsets to vert_offsets
//   Vector<I> vert_offsets = mesh.vf_offsets;
//   for (Size i = 0; i < num_faces; ++i) {
//     auto const & face = mesh.fv[i];
//     for (Size j = 0; j < N; ++j) {
//       auto const vert = static_cast<Size>(face[j]);
//       mesh.vf[static_cast<Size>(vert_offsets[vert])] = static_cast<I>(i);
//       ++vert_offsets[vert];
//     }
//   }
//
// #ifndef NDEBUG
//   // Check for repeated vertices.
//   // This is not technically an error, but it is a sign that the mesh may
//   // cause problems for some algorithms. Hence, we warn the user.
//   auto const bbox = boundingBox(mesh);
//   Vec<D, T> normalization;
//   for (Size i = 0; i < D; ++i) {
//     normalization[i] = static_cast<T>(1) / (bbox.maxima[i] - bbox.minima[i]);
//   }
//   Vector<Point<D, T>> vertices_copy = mesh.vertices;
//   // Transform the points to be in the unit cube
//   for (auto & v : vertices_copy) {
//     v -= bbox.minima;
//     v *= normalization;
//   }
//   if constexpr (std::same_as<T, float>) {
//     mortonSort<uint32_t>(vertices_copy.begin(), vertices_copy.end());
//   } else {
//     mortonSort<uint64_t>(vertices_copy.begin(), vertices_copy.end());
//   }
//   // Revert the scaling
//   for (auto & v : vertices_copy) {
//     v /= normalization;
//   }
//   for (Size i = 0; i < num_vertices - 1; ++i) {
//     if (isApprox(vertices_copy[i], vertices_copy[i + 1])) {
//       Log::warn("Vertex " + std::to_string(i) + " and " + std::to_string(i + 1) +
//                 " are effectively equivalent");
//     }
//   }
//
//   // Check that the vertices are in counter-clockwise order.
//   // If the area of the face is negative, then the vertices are in clockwise
//   for (Size i = 0; i < num_faces; ++i) {
//     if (!mesh.getFace(i).isCCW()) {
//       Log::warn("Face " + std::to_string(i) +
//                 " has vertices in clockwise order. Reordering");
//       mesh.flipFace(i);
//     }
//   }
//
//   // Convexity check
//   // if (file.type == MeshType::Quad) {
//   if constexpr (N == 4) {
//     for (Size i = 0; i < num_faces; ++i) {
//       if (!mesh.getFace(i).isConvex()) {
//         Log::warn("Face " + std::to_string(i) + " is not convex");
//       }
//     }
//   }
// #endif
// }
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// toMeshFile(FaceVertexMesh<P, N, D, T, I> const & mesh, MeshFile<T, I> & file) noexcept
//{
//   // Default to XDMf
//   file.format = MeshFileFormat::XDMF;
//   file.type = getMeshType<P, N>();
//
//   // Vertices
//   if constexpr (D == 3) {
//     file.vertices = mesh.vertices;
//   } else {
//     file.vertices.resize(static_cast<size_t>(mesh.numVertices()));
//     for (Size i = 0; i < mesh.numVertices(); ++i) {
//       file.vertices[static_cast<size_t>(i)][0] = mesh.vertices[i][0];
//       file.vertices[static_cast<size_t>(i)][1] = mesh.vertices[i][1];
//       file.vertices[static_cast<size_t>(i)][2] = 0;
//     }
//   }
//
//   // Faces
//   // NOLINTBEGIN(bugprone-misplaced-widening-cast) justification: It's not misplaced...
//   auto const len = static_cast<size_t>(mesh.numFaces() * N);
//   file.element_conn.resize(len);
//   for (Size i = 0; i < mesh.numFaces(); ++i) {
//     for (Size j = 0; j < N; ++j) {
//       file.element_conn[static_cast<size_t>(i * N + j)] = mesh.fv[i][j];
//     }
//   }
//   // NOLINTEND(bugprone-misplaced-widening-cast)
// }
//
} // namespace um2
