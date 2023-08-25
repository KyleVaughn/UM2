#pragma once

#include <um2.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// Memory management
void
um2Malloc(void ** p, Size size);
void
um2Free(void * p);

// Initialization and finalization
void
um2Initialize(char const * verbosity, Int init_gmsh, Int gmsh_verbosity, Int * ierr);

void
um2Finalize(Int * ierr);

// MPACT Spatial Partition
void
um2NewMPACTSpatialPartition(void ** model, Int * ierr);
void
um2DeleteMPACTSpatialPartition(void * model, Int * ierr);

void
um2ImportMPACTModel(char const * path, void ** model, Int * ierr);

// GetNum
void
um2GetMPACTNumCoarseCells(void * model, Int * n, Int * ierr);
void
um2GetMPACTNumRTMs(void * model, Int * n, Int * ierr);
void
um2GetMPACTNumLattices(void * model, Int * n, Int * ierr);
void
um2GetMPACTNumAssemblies(void * model, Int * n, Int * ierr);

// GetNumCells
void
um2GetMPACTCoreNumCells(void * model, Int * nx, Int * ny, Int * ierr);
void
um2GetMPACTAssemblyNumCells(void * model, Int asy_id, Int * nx, Int * ierr);

// void um2_MPACT_lattice_num_cells(void * const model,
//                                 int const lat_id,
//                                 int * const num_x,
//                                 int * const num_y,
//                                 int * const ierr);
//
// void um2_MPACT_rtm_num_cells(void * const model,
//                              int const rtm_id,
//                              int * const num_x,
//                              int * const num_y,
//                              int * const ierr);
//
// void um2_MPACT_core_get_child(void * const model,
//                              int const i,
//                              int const j,
//                              int * const child,
//                              int * const ierr);
//
// void um2_MPACT_assembly_get_child(void * const model,
//                                  int const id,
//                                  int const i,
//                                  int * const child,
//                                  int * const ierr);
//
// void um2_MPACT_lattice_get_child(void * const model,
//                                 int const id,
//                                 int const i,
//                                 int const j,
//                                 int * const child,
//                                 int * const ierr);
//
// void um2_MPACT_rtm_get_child(void * const model,
//                             int const id,
//                             int const i,
//                             int const j,
//                             int * const child,
//                             int * const ierr);
//
// void um2_MPACT_coarse_cell_num_faces(void * const model,
//                                     int const cc_id,
//                                     int * const num_faces,
//                                     int * const ierr);
//
// void um2_MPACT_coarse_cell_dx(void * const model,
//                              int const cc_id,
//                              double * const dx,
//                              int * const ierr);
//
// void um2_MPACT_coarse_cell_dy(void * const model,
//                              int const cc_id,
//                              double * const dy,
//                              int * const ierr);
//
// void um2_MPACT_coarse_cell_dxdy(void * const model,
//                                int const cc_id,
//                                double * const dx,
//                                double * const dy,
//                                int * const ierr);
//
// void um2_MPACT_coarse_cell_face_areas(void * const model,
//                                      int const cc_id,
//                                      int * const n,         // Number of faces
//                                      double ** const areas,
//                                      int * const ierr);
//
// void um2_MPACT_coarse_cell_find_face(void * const model,
//                                     int const cc_id,
//                                     double const x, // local coordinates
//                                     double const y,
//                                     int * const face_id,
//                                     int * const ierr);
//
// void um2_MPACT_coarse_cell_face_centroid(void * const model,
//                                         int const cc_id,
//                                         int const face_id,
//                                         double * const x, // local coordinates
//                                         double * const y,
//                                         int * const ierr);
//
// void um2_MPACT_coarse_cell_heights(void * const model,
//                                   int * const n,             // Number of heights
//                                   int ** const cc_ids,       // Coarse cell ids array
//                                   ptr double ** const heights,   // Heights array ptr
//                                   int * const ierr);
//
// void um2_MPACT_coarse_cell_material_ids(void * const model,
//                                        int const cc_id,
//                                        MaterialID ** const mat_ids,  // Ptr to first
//                                        mat id int * const n, // Number of mats int *
//                                        const ierr);
//
// void um2_MPACT_module_dimensions(void * const model,
//                                 double * const dx,
//                                 double * const dy,
//                                 double * const dz,
//                                 int * const ierr);
//
// void um2_MPACT_intersect_coarse_cell(void * const model,
//                                     int const cc_id,
//                                     UM2_REAL const origin_x, // pin coordinates
//                                     UM2_REAL const origin_y,
//                                     UM2_REAL const direction_x,
//                                     UM2_REAL const direction_y,
//                                     UM2_REAL * const intersections, // ray iterpolation
//                                     values int * const n,                // in: size of
//                                     array, out: num intersections int * const ierr);
//
//
// void um2_MPACT_rtm_dxdy(void * const model,
//                        int const rtm_id,
//                        double * const dx,
//                        double * const dy,
//                        int * const ierr);
//
// void um2_MPACT_rtm_heights(void * const model,
//                           int * const n,             // Number of heights
//                           int ** const rtm_ids,      // RTM ids array ptr
//                           double ** const heights,   // Heights array ptr
//                           int * const ierr);
//
// void um2_MPACT_lattice_heights(void * const model,
//                               int * const n,             // Number of heights
//                               int ** const lat_ids,      // lattice ids array ptr
//                               double ** const heights,   // Heights array ptr
//                               int * const ierr);
//
// void um2_MPACT_assembly_dzs(void * const model,
//                            int const id,
//                            int * const n,             // Number of heights
//                            double ** const dzs,   // Heights array ptr
//                            int * const ierr);
//
// void um2_MPACT_coarse_cell_face_data(void * const model,
//                                     length_t const cc_id,
//                                     length_t * const mesh_type,
//                                     length_t * const num_vertices,
//                                     length_t * const num_faces,
//                                     UM2_REAL ** const vertices,
//                                     UM2_INT ** const fv_offsets,
//                                     UM2_INT ** const fv,
//                                     int * const ierr);

#ifdef __cplusplus
}
#endif
