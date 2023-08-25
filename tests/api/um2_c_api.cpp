#include <um2c.h>

#include "../test_macros.hpp"

TEST_CASE(malloc_free)
{
  void * ptr = nullptr;
  um2Malloc(&ptr, 4);
  ASSERT(ptr != nullptr);
  um2Free(ptr);
}

TEST_CASE(initialize_finalize)
{
  Int ierr = -1;
  um2Initialize("info", 1, 2, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  um2Finalize(&ierr);
  ASSERT(ierr == 0);
}

TEST_CASE(new_delete_spatial_partition)
{
  Int ierr = -1;
  um2Initialize("warn", 1, 2, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  void * sp = nullptr;
  um2NewMPACTSpatialPartition(&sp, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  um2DeleteMPACTSpatialPartition(sp, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  um2Finalize(&ierr);
  ASSERT(ierr == 0);
}

TEST_CASE(import_MPACT_model)
{
  Int ierr = -1;
  um2Initialize("warn", 1, 2, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  void * sp = nullptr;
  um2ImportMPACTModel("./api_mesh_files/1a.xdmf", &sp, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  auto * const sp_ptr = reinterpret_cast<um2::mpact::SpatialPartition *>(sp);

  // coarse cells
  ASSERT(sp_ptr->coarse_cells.size() == 1);
  Int n = -1;
  um2GetMPACTNumCoarseCells(sp, &n, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  ASSERT(n == 1);
  n = -1;

  ASSERT(sp_ptr->coarse_cells[0].mesh_type == um2::MeshType::Tri);
  ASSERT(sp_ptr->coarse_cells[0].mesh_id == 0);

  ASSERT(sp_ptr->rtms.size() == 1);
  um2GetMPACTNumRTMs(sp, &n, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  ASSERT(n == 1);
  n = -1;

  ASSERT(sp_ptr->lattices.size() == 1);
  um2GetMPACTNumLattices(sp, &n, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  ASSERT(n == 1);
  n = -1;

  ASSERT(sp_ptr->assemblies.size() == 1);
  um2GetMPACTNumAssemblies(sp, &n, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  ASSERT(n == 1);
  n = -1;

  um2DeleteMPACTSpatialPartition(sp, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  um2Finalize(&ierr);
  ASSERT(ierr == 0);
}

TEST_CASE(num_cells)
{
  Int ierr = -1;
  um2Initialize("warn", 1, 2, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  void * sp = nullptr;
  um2ImportMPACTModel("./api_mesh_files/1a.xdmf", &sp, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  Int nx = -1;
  Int ny = -1;

  // Core
  um2GetMPACTCoreNumCells(sp, &nx, &ny, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  ASSERT(nx == 1);
  nx = -1;
  ASSERT(ny == 1);
  ny = -1;

  // Assembly
  um2GetMPACTAssemblyNumCells(sp, 0, &nx, &ierr);
  ASSERT(ierr == 0);
  ierr = -1;
  ASSERT(nx == 1);
  nx = -1;
  //     um2_MPACT_lattice_num_cells(sp, 0, &numx, &numy, &ierr);
  //     ASSERT(ierr == 0, "um2_MPACT_lattice_num_cells"); ierr = -1;
  //     ASSERT(numx == 1, "numx"); numx = -1;
  //     ASSERT(numy == 1, "numy"); numy = -1;
  //     um2_MPACT_rtm_num_cells(sp, 0, &numx, &numy, &ierr);
  //     ASSERT(ierr == 0, "um2_MPACT_rtm_num_cells"); ierr = -1;
  //     ASSERT(numx == 1, "numx"); numx = -1;
  //     ASSERT(numy == 1, "numy"); numy = -1;
  //
  //     um2_delete_MPACT_spatial_partition(sp, &ierr);
  //     ASSERT(ierr == 0, "um2_delete_MPACT_spatial_partition"); ierr = -1;
  //     um2_finalize(&ierr);
  //     ASSERT(ierr == 0, "um2_finalize");
}
//
// TEST(get_child)
//     int ierr = -1;
//     um2_initialize("warn", 1, &ierr);
//     ASSERT(ierr == 0, "um2_initialize"); ierr = -1;
//     void * sp;
//     um2_import_MPACT_model("./test/api/mesh_files/1a.xdmf", &sp, &ierr);
//     ASSERT(ierr == 0, "um2_import_MPACT_model"); ierr = -1;
//     int id = -1;
//     um2_MPACT_core_get_child(sp, 0, 0, &id, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_get_child"); ierr = -1;
//     ASSERT(id == 0, "id"); id = -1;
//     um2_MPACT_assembly_get_child(sp, 0, 0, &id, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_get_child"); ierr = -1;
//     ASSERT(id == 0, "id"); id = -1;
//     um2_MPACT_lattice_get_child(sp, 0, 0, 0, &id, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_get_child"); ierr = -1;
//     ASSERT(id == 0, "id"); id = -1;
//     um2_MPACT_rtm_get_child(sp, 0, 0, 0, &id, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_get_child"); ierr = -1;
//     ASSERT(id == 0, "id"); id = -1;
//
//     um2_delete_MPACT_spatial_partition(sp, &ierr);
//     ASSERT(ierr == 0, "um2_delete_MPACT_spatial_partition"); ierr = -1;
//     um2_finalize(&ierr);
//     ASSERT(ierr == 0, "um2_finalize")
// END_TEST
//
// TEST(coarse_cell_functions)
//     int ierr = -1;
//     um2_initialize("warn", 1, &ierr);
//     ASSERT(ierr == 0, "um2_initialize"); ierr = -1;
//     void * sp;
//     um2_import_MPACT_model("./test/api/mesh_files/1a.xdmf", &sp, &ierr);
//     ASSERT(ierr == 0, "um2_import_MPACT_model"); ierr = -1;
//     // num_faces
//     int n = -1;
//     um2_MPACT_coarse_cell_num_faces(sp, 0, &n, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_num_faces"); ierr = -1;
//     ASSERT(n == 212, "num_faces"); n = -1;
//     // dx, dy
//     double dx = -1;
//     double dy = -1;
//     um2_MPACT_coarse_cell_dx(sp, 0, &dx, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_dx"); ierr = -1;
//     ASSERT_APPROX(dx, 1.26, 1e-4, "dx"); dx = -1;
//     um2_MPACT_coarse_cell_dy(sp, 0, &dy, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_dy"); ierr = -1;
//     ASSERT_APPROX(dy, 1.26, 1e-4, "dy"); dy = -1;
//     um2_MPACT_coarse_cell_dxdy(sp, 0, &dx, &dy, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_dxdy"); ierr = -1;
//     ASSERT_APPROX(dx, 1.26, 1e-4, "dx"); dx = -1;
//     ASSERT_APPROX(dy, 1.26, 1e-4, "dy"); dy = -1;
//
//     um2_MPACT_rtm_dxdy(sp, 0, &dx, &dy, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_rtm_dxdy"); ierr = -1;
//     ASSERT_APPROX(dx, 1.26, 1e-4, "dx"); dx = -1;
//     ASSERT_APPROX(dy, 1.26, 1e-4, "dy"); dy = -1;
//
//     // heights
//     int * cc_ids = nullptr;
//     double * cc_heights = nullptr;
//     um2_MPACT_coarse_cell_heights(sp, &n, &cc_ids, &cc_heights, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_heights"); ierr = -1;
//     ASSERT(n == 1, "n"); n = -1;
//     ASSERT(cc_ids, "cc_ids");
//     ASSERT(cc_heights, "cc_heights");
//     ASSERT(cc_ids[0] == 0, "cc_ids[0]");
//     ASSERT_APPROX(cc_heights[0], 2, 1e-4, "cc_heights[0]");
//     free(cc_ids); cc_ids = nullptr;
//     free(cc_heights); cc_heights = nullptr;
//
//     double * ass_dzs = nullptr;
//     um2_MPACT_assembly_dzs(sp, 0, &n, &ass_dzs, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_assembly_dzs"); ierr = -1;
//     ASSERT(n == 1, "n"); n = -1;
//     ASSERT_APPROX(ass_dzs[0], 2.0, 1e-4, "dz");
//
//     // Coarse cell face areas
//     double * areas = nullptr;
//     um2_MPACT_coarse_cell_face_areas(sp, 0, &n, &areas, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_face_areas"); ierr = -1;
//     ASSERT(n == 212, "n"); n = -1;
//     double area_sum = 0;
//     for (int i = 0; i < 212; ++i) {
//         area_sum += areas[i];
//     }
//     ASSERT_APPROX(area_sum, 1.26*1.26, 1e-4, "area_sum");
//     free(areas); areas = nullptr;
//
//     // material ids
//     MaterialID * mat_ids = nullptr;
//     um2_MPACT_coarse_cell_material_ids(sp, 0, &mat_ids, &n, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_material_ids"); ierr = -1;
//     ASSERT(n == 212, "n"); n = -1;
//     ASSERT(mat_ids, "mat_ids");
//     ASSERT(mat_ids[0] == 1, "mat_ids[0]");
//     ASSERT(mat_ids[1] == 1, "mat_ids[1]");
//
//     // find face
//     int face_id = -2;
//     um2_MPACT_coarse_cell_find_face(sp, 0, 0.1, 0.01, &face_id, &ierr);
//     ASSERT(face_id == 179, "face_id"); face_id = -2;
//     um2_MPACT_coarse_cell_find_face(sp, 0, 0.6, 0.54, &face_id, &ierr);
//     ASSERT(face_id == 0, "face_id"); face_id = -2;
//     um2_MPACT_coarse_cell_find_face(sp, 0, 0.5, -0.05, &face_id, &ierr);
//     ASSERT(face_id == -1, "face_id"); face_id = -2;
//
//     // module_dimensions
//     double dz = 0;
//     dx = 0;
//     dy = 0;
//     um2_MPACT_module_dimensions(sp, &dx, &dy, &dz, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_module_dimensions"); ierr = -1;
//     ASSERT_APPROX(dx, 1.26, 1e-4, "dx"); dx = -1;
//     ASSERT_APPROX(dy, 1.26, 1e-4, "dy"); dy = -1;
//     ASSERT_APPROX(dz, 2, 1e-4, "dz"); dz = -1;
//
//     um2_delete_MPACT_spatial_partition(sp, &ierr);
//     ASSERT(ierr == 0, "um2_delete_MPACT_spatial_partition"); ierr = -1;
//     um2_finalize(&ierr);
//     ASSERT(ierr == 0, "um2_finalize")
// END_TEST
//
// TEST(intersect)
//     int ierr = -1;
//     um2_initialize("warn", 1, &ierr);
//     ASSERT(ierr == 0, "um2_initialize"); ierr = -1;
//     void * sp;
//     // Cheat with some c++
//     um2_MPACT_spatial_partition model;
//     model.make_coarse_cell({1, 1});
//     model.make_coarse_cell({1, 1});
//     model.make_coarse_cell({1, 1});
//     model.make_rtm({{2, 2},
//                     {0, 1}});
//     model.make_lattice({{0}});
//     model.make_assembly({0});
//     model.make_core({{0}});
//     model.import_coarse_cells("./test/MPACT/mesh_files/coarse_cells.inp");
//     sp = &model;
//
//     int const buffer_size = 6;
//     int n = buffer_size;
//     UM2_REAL * buffer = (UM2_REAL *) malloc(sizeof(UM2_REAL) * buffer_size);
//     um2_MPACT_intersect_coarse_cell(sp, 0, 0.0, 0.5, 1.0, 0.0, buffer, &n, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_intersect_coarse_cell"); ierr = -1;
//     ASSERT(n == 4, "n"); n = buffer_size;
//     ASSERT_APPROX(buffer[0], 0.0, 1e-4, "intersections");
//     ASSERT_APPROX(buffer[1], 0.5, 1e-4, "intersections");
//     ASSERT_APPROX(buffer[2], 0.5, 1e-4, "intersections");
//     ASSERT_APPROX(buffer[3], 1.0, 1e-4, "intersections");
//
//     um2_MPACT_intersect_coarse_cell(sp, 1, 0.0, 0.5, 1.0, 0.0, buffer, &n, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_intersect_coarse_cell"); ierr = -1;
//     ASSERT(n == 4, "n"); n = buffer_size;
//     ASSERT_APPROX(buffer[0], 0.0, 1e-4, "intersections");
//     ASSERT_APPROX(buffer[1], 0.5, 1e-4, "intersections");
//     ASSERT_APPROX(buffer[2], 0.5, 1e-4, "intersections");
//     ASSERT_APPROX(buffer[3], 1.0, 1e-4, "intersections");
//
//     n = 0;
//     um2_MPACT_intersect_coarse_cell(sp, 0, 0.0, 0.5, 1.0, 0.0, buffer, &n, &ierr);
//     ASSERT(ierr == 1, "um2_MPACT_intersect_coarse_cell"); ierr = -1;
//     free(buffer);
//
//     length_t mesh_type = -1;
//     length_t nverts = -1;
//     length_t nfaces = -1;
//     UM2_INT * fv_offsets;
//     UM2_INT * fv;
//     UM2_REAL * vertices;
//     um2_MPACT_coarse_cell_face_data(sp, 0, &mesh_type, &nverts, &nfaces,
//                                     &vertices, &fv_offsets, &fv, &ierr);
//     ASSERT(ierr == 0, "um2_MPACT_coarse_cell_face_data"); ierr = -1;
//     ASSERT(mesh_type == 1, "mesh_type"); mesh_type = -1;
//     ASSERT(nverts == 4, "nverts"); nverts = -1;
//     ASSERT(nfaces == 2, "nfaces"); nfaces = -1;
//     ASSERT(!fv_offsets, "fv_offsets");
//     ASSERT(fv, "fv");
//     ASSERT(fv[0] == 0, "fv[0]"); fv[0] = -1;
//     ASSERT(fv[1] == 1, "fv[1]"); fv[1] = -1;
//     ASSERT(fv[2] == 2, "fv[2]"); fv[2] = -1;
//     ASSERT(fv[3] == 2, "fv[3]"); fv[3] = -1;
//     ASSERT(fv[4] == 3, "fv[4]"); fv[4] = -1;
//     ASSERT(fv[5] == 0, "fv[5]"); fv[5] = -1;
//     ASSERT(vertices, "vertices");
//     ASSERT_APPROX(vertices[0], 0.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[1], 0.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[2], 1.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[3], 0.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[4], 1.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[5], 1.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[6], 0.0, 1e-4, "vertices");
//     ASSERT_APPROX(vertices[7], 1.0, 1e-4, "vertices");
//
//     um2_finalize(&ierr);
//     ASSERT(ierr == 0, "um2_finalize")
// END_TEST

TEST_SUITE(c_api)
{
  TEST(malloc_free);
  TEST(initialize_finalize);
  TEST(new_delete_spatial_partition);
  TEST(import_MPACT_model);
  TEST(num_cells);
  //  TEST("get_child", get_child);
  //  TEST("coarse_cell_functions", coarse_cell_functions);
  //  TEST("intersect", intersect);
}

auto
main() -> int
{
  RUN_SUITE(c_api);
  return 0;
}
