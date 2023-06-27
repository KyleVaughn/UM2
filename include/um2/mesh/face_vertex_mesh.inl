#include <um2/geometry/point.hpp>
#include <um2/geometry/polytope.hpp>

#include <thrust/extrema.h> // thrust::minmax_element
#include <thrust/logical.h> // thrust::all_of
#include <thrust/scan.h>    // thrust::inclusive_scan

#include <limits> // std::numeric_limits

namespace um2
{
/*
 template <len_t P, len_t N, std::floating_point T, std::signed_integral I>
 FaceVertexMesh<P, N, T, I>::FaceVertexMesh(MeshFile<T, I> const & file)
{
   // -- Error checking --
   assert(!file.nodes_x.empty());
   assert(!file.nodes_y.empty());
   UM2_ASSERT(!file.element_types.empty());
   UM2_ASSERT(!file.element_offsets.empty());
   UM2_ASSERT(!file.element_conn.empty());

   len_t const num_nodes = file.nodes_x.size();
   len_t const num_elements = file.element_types.size();
   len_t const connectivity_size = file.element_conn.size();

   // Check that all element types are correct
   int8_t type1 = 0;
   int8_t type2 = 0;
   if (file.format == MeshFileFormat::ABAQUS) {
     if constexpr (P == 1 && N == 3) {
       type1 = static_cast<int8_t>(AbaqusCellType::CPS3);
       type2 = static_cast<int8_t>(AbaqusCellType::CPS3);
     } else if constexpr (P == 1 && N == 4) {
       type1 = static_cast<int8_t>(AbaqusCellType::CPS4);
       type2 = static_cast<int8_t>(AbaqusCellType::CPS4);
     } else if constexpr (P == 1 && N == 7) {
       type1 = static_cast<int8_t>(AbaqusCellType::CPS3);
       type2 = static_cast<int8_t>(AbaqusCellType::CPS4);
     } else if constexpr (P == 2 && N == 6) {
       type1 = static_cast<int8_t>(AbaqusCellType::CPS6);
       type2 = static_cast<int8_t>(AbaqusCellType::CPS6);
     } else if constexpr (P == 2 && N == 8) {
       type1 = static_cast<int8_t>(AbaqusCellType::CPS8);
       type2 = static_cast<int8_t>(AbaqusCellType::CPS8);
     } else if constexpr (P == 2 && N == 14) {
       type1 = static_cast<int8_t>(AbaqusCellType::CPS6);
       type2 = static_cast<int8_t>(AbaqusCellType::CPS8);
     } else {
       static_assert(!P, "Unsupported cell type");
     }

   } else if (file.format == MeshFileFormat::XDMF) {
     if constexpr (P == 1 && N == 3) {
       type1 = static_cast<int8_t>(XDMFCellType::TRIANGLE);
       type2 = static_cast<int8_t>(XDMFCellType::TRIANGLE);
     } else if constexpr (P == 1 && N == 4) {
       type1 = static_cast<int8_t>(XDMFCellType::QUAD);
       type2 = static_cast<int8_t>(XDMFCellType::QUAD);
     } else if constexpr (P == 1 && N == 7) {
       type1 = static_cast<int8_t>(XDMFCellType::TRIANGLE);
       type2 = static_cast<int8_t>(XDMFCellType::QUAD);
     } else if constexpr (P == 2 && N == 6) {
       type1 = static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
       type2 = static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
     } else if constexpr (P == 2 && N == 8) {
       type1 = static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
       type2 = static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
     } else if constexpr (P == 2 && N == 14) {
       type1 = static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
       type2 = static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
     } else {
       static_assert(!P, "Unsupported cell type");
     }
   } else {
     UM2_ASSERT(false, "Unsupported mesh format");
   }
   struct TypeCompareFunctor {
     int8_t const type1;
     int8_t const type2;
     __host__ __device__ bool operator()(int8_t type) const {
       return type == type1 || type == type2;
     }
   };
   if (!thrust::all_of(
           file.element_types.cbegin(),
           file.element_types.cend(),
           TypeCompareFunctor{type1, type2})) {
     UM2_ASSERT(false, "Invalid element type");
   }
   // Check that the length of elements is correct by summing the number of
   // elements of each type and the number of vertices of each type.
   // For homogeneous meshes, this is trivial
   bool homogeneous = false;
   if constexpr (P == 1) {
     if constexpr (N == 3 || N == 4) {
       homogeneous = true;
       UM2_ASSERT(connectivity_size == num_elements * N);
     }
   } else if constexpr (P == 2) {
     if constexpr (N == 6 || N == 8) {
       homogeneous = true;
       UM2_ASSERT(connectivity_size == num_elements * N);
     }
   }
   size_t npts1 = 0;
   size_t npts2 = 0;
   if (!homogeneous) {
     // Also check the element offsets
     if (file.format == MeshFileFormat::ABAQUS) {
       if constexpr (P == 1 && N == 3) {
         npts1 = 3;
         npts2 = 3;
       } else if constexpr (P == 1 && N == 4) {
         npts1 = 4;
         npts2 = 4;
       } else if constexpr (P == 1 && N == 7) {
         npts1 = 3;
         npts2 = 4;
       } else if constexpr (P == 2 && N == 6) {
         npts1 = 6;
         npts2 = 6;
       } else if constexpr (P == 2 && N == 8) {
         npts1 = 8;
         npts2 = 8;
       } else if constexpr (P == 2 && N == 14) {
         npts1 = 6;
         npts2 = 8;
       } else {
         static_assert(!P, "Unsupported cell type");
       }
     } else {
       UM2_ASSERT(false, "Unsupported mesh format");
     }
     len_t npts = 0;
     for (len_t i = 0; i < num_elements; ++i) {
       // All elements types have been checked to be valid, so
       // we can use an if-else
       UM2_ASSERT(static_cast<len_t>(file.element_offsets[i]) == npts);
       if (file.element_types[i] == type1) {
         npts += npts1;
       } else if (file.element_types[i] == type2) {
         npts += npts2;
       }
     }
     UM2_ASSERT(npts == connectivity_size);
   }

   // Ensure that each of the vertices has approximately the same z-coordinate
   if (!file.nodes_z.empty()) {
     T const z = file.nodes_z[0];
     for (len_t i = 1; i < num_nodes; ++i) {
       UM2_ASSERT(std::abs(file.nodes_z[i] - z) < EPS_POINT<T>);
     }
   }

   // -- vertices --

   this->vertices.resize(num_nodes);
   for (len_t i = 0; i < num_nodes; ++i) {
     this->vertices[i] = {file.nodes_x[i], file.nodes_y[i]};
     UM2_ASSERT(file.nodes_x[i] >= 0, "Negative x-coordinate");
     UM2_ASSERT(file.nodes_y[i] >= 0, "Negative y-coordinate");
   }

   // -- fv/fv_offsets --

   this->fv = file.element_conn;
   if (!homogeneous) {
     this->fv_offsets = file.element_offsets;
   }

   // -- vf/vf_offsets --

   Vector<I> vert_counts(num_nodes, 0);
   for (len_t i = 0; i < connectivity_size; ++i) {
     ++vert_counts[file.element_conn[i]];
   }
   this->vf_offsets.resize(num_nodes + 1);
   this->vf_offsets[0] = 0;
   thrust::inclusive_scan(vert_counts.cbegin(),
                          vert_counts.cend(),
                          vf_offsets.begin() + 1);
   vert_counts.clear();

   this->vf.resize(this->vf_offsets[num_nodes]);
   // Copy vf_offsets to vert_offsets
   Vector<I> vert_offsets;
   vert_offsets = this->vf_offsets;
   if (homogeneous) {
     for (I i = 0; i < num_elements; ++i) {
       for (I j = i * N; j < (i + 1) * N; ++j) {
         I const vert = this->fv[j];
         this->vf[vert_offsets[vert]] = i;
         ++vert_offsets[vert];
       }
     }
   } else {
     for (I i = 0; i < num_elements; ++i) {
       I const face_start = this->fv_offsets[i    ];
       I const face_end =   this->fv_offsets[i + 1];
       for (I j = face_start; j < face_end; ++j) {
         I const vert = this->fv[j];
         this->vf[vert_offsets[vert]] = i;
         ++vert_offsets[vert];
       }
     }
   }
 };
*/
// -- Methods --

template <len_t N, std::floating_point T, std::signed_integral I>
UM2_NDEBUG_PURE auto
boundingBox(LinearPolygonMesh<N, T, I> const & mesh) -> AABox2<T>
{
  return boundingBox(mesh.vertices);
}

// template <len_t N, std::floating_point T, std::signed_integral I>
// UM2_NDEBUG_PURE UM2_HOSTDEV AABox2<T>
// bounding_box(QuadraticPolygonMesh<N, T, I> const & mesh)
//{
//   // Get the bounding box that bounds every edge.
//   AABox2<T> box;
//   box.minima = Point2<T>(std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
//   box.maxima =
//       Point2<T>(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest());
//   // If the mesh is homogeneous, then offsets are not used
//   if constexpr (N == 6 || N == 8) {
//     constexpr len_t M = N >> 1;
//     for (len_t iface = 0; iface < num_faces(mesh); ++iface) {
//       len_t const offset = iface * N;
//       for (len_t iedge = 0; iedge + 1 < M; ++iedge) {
//         I const v0 = mesh.fv[offset + iedge];
//         I const v1 = mesh.fv[offset + iedge + 1];
//         I const v2 = mesh.fv[offset + iedge + M];
//         Point2<T> const & p0 = mesh.vertices[v0];
//         Point2<T> const & p1 = mesh.vertices[v1];
//         Point2<T> const & p2 = mesh.vertices[v2];
//         AABox2<T> const edge_box = quadratic_segment_bounding_box(p0, p1, p2);
//         box = bounding_box(box, edge_box);
//       }
//       I const v0 = mesh.fv[offset + M - 1];
//       I const v1 = mesh.fv[offset];
//       I const v2 = mesh.fv[offset + N - 1];
//       Point2<T> const & p0 = mesh.vertices[v0];
//       Point2<T> const & p1 = mesh.vertices[v1];
//       Point2<T> const & p2 = mesh.vertices[v2];
//       AABox2<T> const edge_box = quadratic_segment_bounding_box(p0, p1, p2);
//       box = bounding_box(box, edge_box);
//     }
//   } else if constexpr (N == 14) {
//     for (len_t iface = 0; iface < num_faces(mesh); ++iface) {
//       I const low_offset = mesh.fv_offsets[iface];
//       I const high_offset = mesh.fv_offsets[iface + 1];
//       I const n = high_offset - low_offset;
//       I const M = n >> 1;
//       for (I iedge = 0; iedge + 1 < M; ++iedge) {
//         I const v0 = mesh.fv[low_offset + iedge];
//         I const v1 = mesh.fv[low_offset + iedge + 1];
//         I const v2 = mesh.fv[low_offset + iedge + M];
//         Point2<T> const & p0 = mesh.vertices[v0];
//         Point2<T> const & p1 = mesh.vertices[v1];
//         Point2<T> const & p2 = mesh.vertices[v2];
//         AABox2<T> const edge_box = quadratic_segment_bounding_box(p0, p1, p2);
//         box = bounding_box(box, edge_box);
//       }
//       I const v0 = mesh.fv[low_offset + M - 1];
//       I const v1 = mesh.fv[low_offset];
//       I const v2 = mesh.fv[low_offset + n - 1];
//       Point2<T> const & p0 = mesh.vertices[v0];
//       Point2<T> const & p1 = mesh.vertices[v1];
//       Point2<T> const & p2 = mesh.vertices[v2];
//       AABox2<T> const edge_box = quadratic_segment_bounding_box(p0, p1, p2);
//       box = bounding_box(box, edge_box);
//     }
//   } else {
//     static_assert(!N, "Unsupported mesh type");
//   }
//   return box;
// }

// template <len_t P, len_t N, std::floating_point T, std::signed_integral I>
// void FaceVertexMesh<P, N, T, I>::to_mesh_file(MeshFile<T, I> & file) const
//{
//   // Default to xdmf format
//   file.format = MeshFileFormat::XDMF;
//
//   file.nodes_x.resize(this->vertices.size());
//   file.nodes_y.resize(this->vertices.size());
//   file.nodes_z.resize(this->vertices.size());
//   for (I i = 0; i < this->vertices.size(); ++i) {
//     file.nodes_x[i] = this->vertices[i][0];
//     file.nodes_y[i] = this->vertices[i][1];
//     file.nodes_z[i] = 0;
//   }
//   len_t const nfaces = num_faces(*this);
//   file.element_types.resize(nfaces);
//   file.element_offsets.resize(nfaces + 1);
//   file.element_offsets[0] = 0;
//   if constexpr (P == 1 && N == 3) {
//     file.element_conn.resize(nfaces * N);
//     for (len_t i = 0; i < nfaces; ++i) {
//       file.element_types[i] = static_cast<int8_t>(XDMFCellType::TRIANGLE);
//       file.element_offsets[i + 1] = (i + 1) * N;
//       for (len_t j = 0; j < N; ++j) {
//         file.element_conn[i * N + j] = this->fv[i * N + j];
//       }
//     }
//   } else if constexpr (P == 1 && N == 4) {
//     file.element_conn.resize(nfaces * N);
//     for (len_t i = 0; i < nfaces; ++i) {
//       file.element_types[i] = static_cast<int8_t>(XDMFCellType::QUAD);
//       file.element_offsets[i + 1] = (i + 1) * N;
//       for (len_t j = 0; j < N; ++j) {
//         file.element_conn[i * N + j] = this->fv[i * N + j];
//       }
//     }
//   } else if constexpr (P == 2 && N == 6) {
//     file.element_conn.resize(nfaces * N);
//     for (len_t i = 0; i < nfaces; ++i) {
//       file.element_types[i] = static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
//       file.element_offsets[i + 1] = (i + 1) * N;
//       for (len_t j = 0; j < N; ++j) {
//         file.element_conn[i * N + j] = this->fv[i * N + j];
//       }
//     }
//   } else if constexpr (P == 2 && N == 8) {
//     file.element_conn.resize(nfaces * N);
//     for (len_t i = 0; i < nfaces; ++i) {
//       file.element_types[i] = static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
//       file.element_offsets[i + 1] = (i + 1) * N;
//       for (len_t j = 0; j < N; ++j) {
//         file.element_conn[i * N + j] = this->fv[i * N + j];
//       }
//     }
//   } else if constexpr (P == 1 && N == 7) {
//     for (len_t i = 0; i < nfaces; ++i) {
//       I const face_start = this->fv_offsets[i];
//       I const face_end = this->fv_offsets[i + 1];
//       I const face_size = face_end - face_start;
//       if (face_size == 3) {
//         file.element_types[i] = static_cast<int8_t>(XDMFCellType::TRIANGLE);
//       } else if (face_size == 4) {
//         file.element_types[i] = static_cast<int8_t>(XDMFCellType::QUAD);
//       } else {
//         Log::error("Unsupported face size");
//       }
//       file.element_offsets[i + 1] = file.element_offsets[i] + face_size;
//     }
//     file.element_conn = this->fv;
//   } else if constexpr (P == 2 && N == 14) {
//     for (len_t i = 0; i < nfaces; ++i) {
//       I const face_start = this->fv_offsets[i];
//       I const face_end = this->fv_offsets[i + 1];
//       I const face_size = face_end - face_start;
//       if (face_size == 6) {
//         file.element_types[i] = static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
//       } else if (face_size == 8) {
//         file.element_types[i] = static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
//       } else {
//         Log::error("Unsupported face size");
//       }
//       file.element_offsets[i + 1] = file.element_offsets[i] + face_size;
//     }
//     file.element_conn = this->fv;
//   } else {
//     static_assert(!P, "Unsupported mesh type");
//   }
// }

////// -- IO --
////
////template <len_t K, len_t P, len_t N, len_t D, typename T>
////std::ostream & operator << (std::ostream &, Polytope<K, P, N, D, T> const &);
//
} // namespace um2