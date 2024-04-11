#pragma once

#include <um2.hpp>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Data sizes 
//==============================================================================

void
um2SizeOfInt(int * size);

void
um2SizeOfFloat(int * size);

//==============================================================================
// Memory management
//==============================================================================

void
um2Malloc(void ** p, Int size);

void
um2Free(void * p);

//==============================================================================
// Initialization and finalization
//==============================================================================

void
um2Initialize();

void
um2Finalize();

//// MPACT Spatial Partition
//void
//um2NewMPACTSpatialPartition(void ** model, Int * ierr);
//void
//um2DeleteMPACTSpatialPartition(void * model, Int * ierr);
//
//void
//um2ImportMPACTModel(char const * path, void ** model, Int * ierr);
//
//// Num
//void
//um2MPACTNumCoarseCells(void * model, Int * n, Int * ierr);
//void
//um2MPACTNumRTMs(void * model, Int * n, Int * ierr);
//void
//um2MPACTNumLattices(void * model, Int * n, Int * ierr);
//void
//um2MPACTNumAssemblies(void * model, Int * n, Int * ierr);
//
//// NumCells
//void
//um2MPACTCoreNumCells(void * model, Int * nx, Int * ny, Int * ierr);
//void
//um2MPACTAssemblyNumCells(void * model, Int asy_id, Int * nx, Int * ierr);
//void
//um2MPACTLatticeNumCells(void * model, Int lat_id, Int * nx, Int * ny, Int * ierr);
//void
//um2MPACTRTMNumCells(void * model, Int rtm_id, Int * nx, Int * ny, Int * ierr);
//
//// Child
//void
//um2MPACTCoreGetChild(void * model, Int ix, Int iy, Int * child, Int * ierr);
//void
//um2MPACTAssemblyGetChild(void * model, Int asy_id, Int ix, Int * child, Int * ierr);
//void
//um2MPACTLatticeGetChild(void * model, Int lat_id, Int ix, Int iy, Int * child,
//                        Int * ierr);
//void
//um2MPACTRTMGetChild(void * model, Int rtm_id, Int ix, Int iy, Int * child, Int * ierr);
//
//// CoarseCell
//void
//um2MPACTCoarseCellNumFaces(void * model, Int cc_id, Int * num_faces, Int * ierr);
//void
//um2MPACTCoarseCellWidth(void * model, Int cc_id, Float * width, Int * ierr);
//void
//um2MPACTCoarseCellHeight(void * model, Int cc_id, Float * height, Int * ierr);
//void
//um2MPACTCoarseCellFaceAreas(void * model, Int cc_id, Int * n, Float ** areas, Int * ierr);
//void
//um2MPACTCoarseCellFaceContaining(void * model, Int cc_id, Float x, Float y, Int * face_id,
//                                 Int * ierr);
//void
//um2MPACTCoarseCellFaceCentroid(void * model, Int cc_id, Int face_id, Float * x, Float * y,
//                               Int * ierr);
//void
//um2MPACTCoarseCellMaterialIDs(void * model, Int cc_id, MaterialID ** mat_ids, Int * n,
//                              Int * ierr);
//void
//um2MPACTIntersectCoarseCell(void * model, Int cc_id, Float origin_x, Float origin_y,
//                            Float direction_x, Float direction_y, Float * intersections,
//                            Int * n, Int * ierr);
//void
//um2MPACTRTMWidth(void * model, Int rtm_id, Float * width, Int * ierr);
//void
//um2MPACTRTMHeight(void * model, Int rtm_id, Float * height, Int * ierr);
//void
//um2MPACTCoarseCellHeights(void * model, Int * n, Int ** cc_ids, Float ** heights,
//                          Int * ierr);
//void
//um2MPACTRTMHeights(void * model, Int * n, Int ** rtm_ids, Float ** heights, Int * ierr);
//void
//um2MPACTLatticeHeights(void * model, Int * n, Int ** lat_ids, Float ** heights,
//                       Int * ierr);
//void
//um2MPACTAssemblyHeights(void * model, Int asy_id, Int * n, Float ** heights, Int * ierr);
//
//void
//um2MPACTCoarseCellFaceData(void * model, Int cc_id, Int * mesh_type, Int * num_vertices,
//                           Int * num_faces, Float ** vertices, Int ** fv, Int * ierr);

#ifdef __cplusplus
}
#endif
