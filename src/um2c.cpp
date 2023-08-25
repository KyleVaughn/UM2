#include <um2c.h>

#define TRY_CATCH(func)                                                                  \
  if (ierr != nullptr) {                                                                 \
    *ierr = 0;                                                                           \
  }                                                                                      \
  try {                                                                                  \
    func;                                                                                \
  } catch (...) {                                                                        \
    if (ierr != nullptr) {                                                               \
      *ierr = 1;                                                                         \
    }                                                                                    \
  }

//==============================================================================
// um2Malloc and um2Free
//==============================================================================

void
um2Malloc(void ** const p, Size const size)
{
  *p = malloc(static_cast<size_t>(size));
}

void
um2Free(void * const p)
{
  if (p != nullptr) {
    free(p);
  }
}

//==============================================================================
// um2Initialize and um2Finalize
//==============================================================================

void
um2Initialize(char const * const verbosity, Int const init_gmsh, Int const gmsh_verbosity,
              Int * const ierr)
{
  TRY_CATCH(um2::initialize(verbosity, init_gmsh == 1, gmsh_verbosity));
}

void
um2Finalize(Int * const ierr)
{
  TRY_CATCH(um2::finalize());
}

//==============================================================================
// MPACT SpatialPartition
//==============================================================================

void
um2NewMPACTSpatialPartition(void ** const model, Int * const ierr)
{
  TRY_CATCH(*model = reinterpret_cast<void *>(new um2::mpact::SpatialPartition()));
}

void
um2DeleteMPACTSpatialPartition(void * const model, Int * const ierr)
{
  TRY_CATCH(delete reinterpret_cast<um2::mpact::SpatialPartition *>(model));
}

void
um2ImportMPACTModel(char const * const path, void ** const model, Int * const ierr)
{
  TRY_CATCH({
    std::string const path_str(path);
    *model = reinterpret_cast<void *>(new um2::mpact::SpatialPartition());
    um2::importMesh(path_str, *reinterpret_cast<um2::mpact::SpatialPartition *>(*model));
  });
}

void
um2GetMPACTNumCoarseCells(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.coarse_cells.size();
  });
}

void
um2GetMPACTNumRTMs(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.rtms.size();
  });
}

void
um2GetMPACTNumLattices(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.lattices.size();
  });
}

void
um2GetMPACTNumAssemblies(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.assemblies.size();
  });
}

void
um2GetMPACTCoreNumCells(void * const model, int * const nx, int * const ny,
                        int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const ncells = sp.core.numCells();
    *nx = ncells[0];
    *ny = ncells[1];
  });
}

void
um2GetMPACTAssemblyNumCells(void * const model, Int const asy_id, Int * const nx,
                            Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *nx = sp.assemblies[asy_id].numXCells();
  });
}
//
// void um2_MPACT_lattice_num_cells(void * const model,
//                                  int const lat_id,
//                                  int * const num_x,
//                                  int * const num_y,
//                                  int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *num_x = static_cast<int>(um2::num_xcells(sp->lattices[lat_id]));
//         *num_y = static_cast<int>(um2::num_ycells(sp->lattices[lat_id]));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_rtm_num_cells(void * const model,
//                                  int const rtm_id,
//                                  int * const num_x,
//                                  int * const num_y,
//                                  int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *num_x = static_cast<int>(um2::num_xcells(sp->rtms[rtm_id]));
//         *num_y = static_cast<int>(um2::num_ycells(sp->rtms[rtm_id]));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_core_get_child(void * const model,
//                               int const i,
//                               int const j,
//                               int * const child,
//                               int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *child = static_cast<int>(sp->core.get_child(static_cast<length_t>(i),
//                                                      static_cast<length_t>(j)));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_assembly_get_child(void * const model,
//                                   int const id,
//                                   int const i,
//                                   int * const child,
//                                   int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *child =
//         static_cast<int>(sp->assemblies[id].get_child(static_cast<length_t>(i)));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_lattice_get_child(void * const model,
//                                   int const id,
//                                   int const i,
//                                   int const j,
//                                   int * const child,
//                                   int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *child = static_cast<int>(sp->lattices[id].get_child(static_cast<length_t>(i),
//                                                              static_cast<length_t>(j)));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_rtm_get_child(void * const model,
//                                   int const id,
//                                   int const i,
//                                   int const j,
//                                   int * const child,
//                                   int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *child = static_cast<int>(sp->rtms[id].get_child(static_cast<length_t>(i),
//                                                          static_cast<length_t>(j)));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_num_faces(void * const model,
//                                      int const cc_id,
//                                      int * const num_faces,
//                                      int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *num_faces = static_cast<int>(sp->coarse_cells[cc_id].num_faces());
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_dx(void * const model,
//                               int const cc_id,
//                               double * const dx,
//                               int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *dx = static_cast<double>(sp->coarse_cells[cc_id].dxdy[0]);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_dy(void * const model,
//                               int const cc_id,
//                               double * const dy,
//                               int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *dy = static_cast<double>(sp->coarse_cells[cc_id].dxdy[1]);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_dxdy(void * const model,
//                                 int const cc_id,
//                                 double * const dx,
//                                 double * const dy,
//                                 int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *dx = static_cast<double>(sp->coarse_cells[cc_id].dxdy[0]);
//         *dy = static_cast<double>(sp->coarse_cells[cc_id].dxdy[1]);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_face_areas(void * const model,
//                                       int const cc_id,
//                                       int * const n,         // Number of faces
//                                       double ** const areas,
//                                       int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         um2::Vector<UM2_REAL> vareas;
//         sp->coarse_cell_face_areas(cc_id, vareas);
//         *n = static_cast<int>(vareas.size());
//         *areas = (double *)malloc(*n * sizeof(double));
//         for (int i = 0; i < *n; ++i) {
//             (*areas)[i] = static_cast<double>(vareas[i]);
//         }
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_find_face(void * const model,
//                                      int const cc_id,
//                                      double const x,
//                                      double const y,
//                                      int * const face_id,
//                                      int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         UM2_REAL const xt = static_cast<UM2_REAL>(x);
//         UM2_REAL const yt = static_cast<UM2_REAL>(y);
//         length_t const id = sp->coarse_cell_find_face(cc_id,
//                                         um2::Point2<UM2_REAL>(xt, yt));
//         *face_id = static_cast<int>(id);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_face_centroid(void * const model,
//                                          int const cc_id,
//                                          int const face_id,
//                                          double * const x, // local coordinates
//                                          double * const y,
//                                          int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         um2::Point2<UM2_REAL> const pt = sp->coarse_cell_face_centroid(cc_id, face_id);
//         *x = static_cast<double>(pt[0]);
//         *y = static_cast<double>(pt[1]);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_heights(void * const model,
//                                    int * const n,       // Number of heights
//                                    int ** const cc_ids, // Coarse cell ids array ptr
//                                    double ** heights,   // Heights array ptr
//                                    int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//             reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         um2::Vector<std::pair<int, double>> id_dz;
//         sp->coarse_cell_heights(id_dz);
//         *n = static_cast<int>(id_dz.size());
//         *cc_ids = (int *)malloc(*n * sizeof(int));
//         *heights = (double *)malloc(*n * sizeof(double));
//         for (int i = 0; i < *n; ++i) {
//             (*cc_ids)[i] = id_dz[i].first;
//             (*heights)[i] = id_dz[i].second;
//         }
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_material_ids(void * const model,
//                                         int const cc_id,
//                                         MaterialID ** const mat_ids,  // Ptr to first
//                                         mat id int * const n, // Number of mats int *
//                                         const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         *n = static_cast<int>(sp->coarse_cells[cc_id].material_ids.size());
//         *mat_ids = &sp->coarse_cells[cc_id].material_ids[0];
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_module_dimensions(void * const model,
//                                  double * const dx,
//                                  double * const dy,
//                                  double * const dz,
//                                  int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         auto const aabb2 = bounding_box(sp->rtms[0]);
//         *dx = static_cast<double>(um2::width(aabb2));
//         *dy = static_cast<double>(um2::height(aabb2));
//         auto const aabb1 = sp->assemblies[0].get_box(0);
//         *dz = static_cast<double>(um2::width(aabb1));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
//
// }
//
// void um2_MPACT_intersect_coarse_cell(void * const model,
//                                      int const cc_id,
//                                      UM2_REAL const origin_x, // pin coordinates
//                                      UM2_REAL const origin_y,
//                                      UM2_REAL const direction_x,
//                                      UM2_REAL const direction_y,
//                                      UM2_REAL * const intersections, // ray
//                                      iterpolation values int * const n, // number of
//                                      intersections int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         if (*n <= 0) {
//             *ierr = 1;
//             return;
//         }
//         um2_MPACT_spatial_partition * const sp =
//             reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         um2::Ray2<UM2_REAL> const ray(um2::Point2<UM2_REAL>(origin_x, origin_y),
//                                       um2::Vec2<UM2_REAL>(direction_x, direction_y));
//         sp->intersect_coarse_cell(cc_id, ray, intersections, n);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_rtm_dxdy(void * const model,
//                         int const rtm_id,
//                         double * const dx,
//                         double * const dy,
//                         int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//           reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         auto const aabb2 = um2::bounding_box(sp->rtms[rtm_id]);
//         *dx = static_cast<double>(width(aabb2));
//         *dy = static_cast<double>(height(aabb2));
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_rtm_heights(void * const model,
//                            int * const n,             // Number of heights
//                            int ** const rtm_ids,      // RTM ids array ptr
//                            double ** const heights,   // Heights array ptr
//                            int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//             reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         um2::Vector<std::pair<int, double>> id_dz;
//         sp->rtm_heights(id_dz);
//         *n = static_cast<int>(id_dz.size());
//         *rtm_ids = (int *)malloc(*n * sizeof(int));
//         *heights = (double *)malloc(*n * sizeof(double));
//         for (int i = 0; i < *n; ++i) {
//             (*rtm_ids)[i] = id_dz[i].first;
//             (*heights)[i] = id_dz[i].second;
//         }
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_lattice_heights(void * const model,
//                                int * const n,             // Number of heights
//                                int ** const lat_ids,      // lat ids array ptr
//                                double ** const heights,   // Heights array ptr
//                                int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//             reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         um2::Vector<std::pair<int, double>> id_dz;
//         sp->lattice_heights(id_dz);
//         *n = static_cast<int>(id_dz.size());
//         *lat_ids = (int *)malloc(*n * sizeof(int));
//         *heights = (double *)malloc(*n * sizeof(double));
//         for (int i = 0; i < *n; ++i) {
//             (*lat_ids)[i] = id_dz[i].first;
//             (*heights)[i] = id_dz[i].second;
//         }
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_assembly_dzs(void * const model,
//                             int const id,
//                             int * const n,             // Number of heights
//                             double ** const dzs,   // Heights array ptr
//                             int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//             reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         length_t const n_dz = sp->assemblies[id].children.size();
//         *n = static_cast<int>(n_dz);
//         *dzs = (double *)malloc(*n * sizeof(double));
//         for (int i = 0; i < *n; ++i) {
//             (*dzs)[i] = static_cast<double>(um2::width(sp->assemblies[id].get_box(i)));
//         }
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
//
// void um2_MPACT_coarse_cell_face_data(void * const model,
//                                      length_t const cc_id,
//                                      length_t * const mesh_type,
//                                      length_t * const num_vertices,
//                                      length_t * const num_faces,
//                                      UM2_REAL ** const vertices,
//                                      UM2_INT ** const fv_offsets,
//                                      UM2_INT ** const fv,
//                                      int * const ierr)
//{
//     if (ierr) *ierr = 0;
//     try {
//         um2_MPACT_spatial_partition * const sp =
//             reinterpret_cast<um2_MPACT_spatial_partition *>(model);
//         sp->coarse_cell_face_data(cc_id, mesh_type, num_vertices, num_faces, vertices,
//         fv_offsets, fv);
//     } catch (...) {
//         if (ierr) *ierr = 1;
//     }
// }
