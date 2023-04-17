#include <um2c.h>
#include <cstdlib>

#include <thrust/sort.h>

void um2_malloc(void ** const p, int const size)
{
    *p = malloc(size);
}

void um2_free(void * const p)
{
    if (p) free(p);
}

void um2_sort(double * begin, double * end)
{
    thrust::sort(begin, end);
}

//void um2_initialize(char const * const verbosity, 
//                    int const init_gmsh, 
//                    int * const ierr)
//{
//    if (ierr) *ierr = 0;
//    try {
//        um2::initialize(verbosity, init_gmsh);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_finalize(int * const ierr)
//{
//    if (ierr) *ierr = 0;
//    try {
//        um2::finalize();
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_new_mpact_spatial_partition(void ** const model, int * const ierr)
//{
//    if (ierr) *ierr = 0;
//    try {
//        *model = reinterpret_cast<void *>(new um2_mpact_spatial_partition());
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_delete_mpact_spatial_partition(void * const model, int * const ierr)
//{
//    if (ierr) *ierr = 0;
//    try {
//        delete reinterpret_cast<um2_mpact_spatial_partition *>(model);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_import_mpact_model(char const * const path,    
//                            void ** const model, 
//                            int * const ierr)
//{
//    if (ierr) *ierr = 0;
//    try {
//        std::string const path_str(path);
//        *model = reinterpret_cast<void *>(new um2_mpact_spatial_partition());
//        um2::import_mpact_model(path, 
//                                *reinterpret_cast<um2_mpact_spatial_partition *>(*model)); 
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//    
//void um2_mpact_num_unique_coarse_cells(void * const model, int * const n, int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        *n = um2::mpact::num_unique_coarse_cells(
//          *reinterpret_cast<um2_mpact_spatial_partition *>(model));
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_num_unique_rtms(void * const model, int * const n, int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        *n = um2::mpact::num_unique_rtms(
//          *reinterpret_cast<um2_mpact_spatial_partition *>(model));
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_num_unique_lattices(void * const model, int * const n, int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        *n = um2::mpact::num_unique_lattices(
//          *reinterpret_cast<um2_mpact_spatial_partition *>(model));
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_num_unique_assemblies(void * const model, int * const n, int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        *n = um2::mpact::num_unique_assemblies(
//          *reinterpret_cast<um2_mpact_spatial_partition *>(model));
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_core_num_cells(void * const model, 
//                              int * const num_x, 
//                              int * const num_y,
//                              int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        *num_x = static_cast<int>(um2::num_xcells(sp->core));
//        *num_y = static_cast<int>(um2::num_ycells(sp->core));
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_core_get_child(void * const model,    
//                              int const i,    
//                              int const j,    
//                              int * const child, 
//                              int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        *child = static_cast<int>(sp->core.get_child(static_cast<length_t>(i),
//                                                     static_cast<length_t>(j)));
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_num_faces(void * const model,    
//                                     int const cc_id,     
//                                     int * const num_faces,    
//                                     int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        *num_faces = static_cast<int>(sp->coarse_cells[cc_id].num_faces());
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_dx(void * const model,    
//                              int const cc_id,     
//                              double * const dx,
//                              int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        *dx = static_cast<double>(sp->coarse_cells[cc_id].dxdy[0]);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_dy(void * const model,    
//                              int const cc_id,     
//                              double * const dy,
//                              int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        *dy = static_cast<double>(sp->coarse_cells[cc_id].dxdy[1]);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_dxdy(void * const model,    
//                                int const cc_id,     
//                                double * const dx,
//                                double * const dy,
//                                int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        *dx = static_cast<double>(sp->coarse_cells[cc_id].dxdy[0]);
//        *dy = static_cast<double>(sp->coarse_cells[cc_id].dxdy[1]);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_face_areas(void * const model,    
//                                      int const cc_id,
//                                      int * const n,         // Number of faces           
//                                      double ** const areas,                             
//                                      int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        um2::Vector<UM2_REAL> vareas;
//        sp->coarse_cell_face_areas(cc_id, vareas);
//        *n = static_cast<int>(vareas.size());
//        *areas = (double *)malloc(*n * sizeof(double));
//        for (int i = 0; i < *n; ++i) {
//            (*areas)[i] = static_cast<double>(vareas[i]);
//        }
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_find_face(void * const model,
//                                     int const cc_id, 
//                                     double const x,
//                                     double const y,
//                                     int * const face_id,   
//                                     int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        UM2_REAL const xt = static_cast<UM2_REAL>(x);
//        UM2_REAL const yt = static_cast<UM2_REAL>(y);
//        length_t const id = sp->coarse_cell_find_face(cc_id, 
//                                        um2::Point2<UM2_REAL>(xt, yt));
//        *face_id = static_cast<int>(id);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_face_centroid(void * const model,
//                                         int const cc_id,   
//                                         int const face_id,
//                                         double * const x, // local coordinates 
//                                         double * const y,
//                                         int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//          reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        um2::Point2<UM2_REAL> const pt = sp->coarse_cell_face_centroid(cc_id, face_id);
//        *x = static_cast<double>(pt[0]);
//        *y = static_cast<double>(pt[1]);
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
//void um2_mpact_coarse_cell_heights(void * const model,
//                                   int * const n,       // Number of heights 
//                                   int ** const cc_ids, // Coarse cell ids array ptr
//                                   double ** heights,   // Heights array ptr
//                                   int * const ierr)
//{
//    if (ierr) *ierr = 0;                                                            
//    try {
//        um2_mpact_spatial_partition * const sp = 
//            reinterpret_cast<um2_mpact_spatial_partition *>(model);
//        um2::Vector<std::pair<int, double>> id_dz;
//        sp->coarse_cell_heights(id_dz);
//        *n = static_cast<int>(id_dz.size());
//        *cc_ids = (int *)malloc(*n * sizeof(int)); 
//        *heights = (double *)malloc(*n * sizeof(double)); 
//        for (int i = 0; i < *n; ++i) {
//            (*cc_ids)[i] = id_dz[i].first;
//            (*heights)[i] = id_dz[i].second;
//        }
//    } catch (...) {
//        if (ierr) *ierr = 1;
//    }
//}
//
