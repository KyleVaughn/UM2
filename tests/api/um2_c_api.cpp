#include <um2c.h>

#include "../test_macros.hpp"

TEST_CASE(data_sizes)
{
  int n = -1;
  um2SizeOfInt(&n);
  ASSERT(n == sizeof(Int)); 
  n = -1;
  um2SizeOfFloat(&n);
  ASSERT(n == sizeof(Float));
}

TEST_CASE(malloc_free)
{
  void * ptr = nullptr;
  um2Malloc(&ptr, 4);
  ASSERT(ptr != nullptr);
  um2Free(ptr);
}

TEST_CASE(initialize_finalize)
{
  um2Initialize();
  um2Finalize();
}

//TEST_CASE(new_delete_spatial_partition)
//{
//  Int ierr = -1;
//  um2Initialize("warn", 1, 2, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  void * sp = nullptr;
//  um2NewMPACTSpatialPartition(&sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  um2DeleteMPACTSpatialPartition(sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  um2Finalize(&ierr);
//  ASSERT(ierr == 0);
//}
//
//TEST_CASE(import_MPACT_model)
//{
//  Int ierr = -1;
//  um2Initialize("warn", 1, 2, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  void * sp = nullptr;
//  um2ImportMPACTModel("./api_mesh_files/1a.xdmf", &sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  auto * const sp_ptr = reinterpret_cast<um2::mpact::SpatialPartition *>(sp);
//
//  // coarse cells
//  ASSERT(sp_ptr->coarse_cells.size() == 1);
//  Int n = -1;
//  um2MPACTNumCoarseCells(sp, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 1);
//  n = -1;
//
//  ASSERT(sp_ptr->coarse_cells[0].mesh_type == um2::MeshType::Tri);
//  ASSERT(sp_ptr->coarse_cells[0].mesh_id == 0);
//
//  ASSERT(sp_ptr->rtms.size() == 1);
//  um2MPACTNumRTMs(sp, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 1);
//  n = -1;
//
//  ASSERT(sp_ptr->lattices.size() == 1);
//  um2MPACTNumLattices(sp, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 1);
//  n = -1;
//
//  ASSERT(sp_ptr->assemblies.size() == 1);
//  um2MPACTNumAssemblies(sp, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 1);
//  n = -1;
//
//  um2DeleteMPACTSpatialPartition(sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  um2Finalize(&ierr);
//  ASSERT(ierr == 0);
//}
//
//TEST_CASE(mpact_num_cells)
//{
//  Int ierr = -1;
//  um2Initialize("warn", 1, 2, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  void * sp = nullptr;
//  um2ImportMPACTModel("./api_mesh_files/1a.xdmf", &sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  Int nx = -1;
//  Int ny = -1;
//
//  // Core
//  um2MPACTCoreNumCells(sp, &nx, &ny, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(nx == 1);
//  nx = -1;
//  ASSERT(ny == 1);
//  ny = -1;
//
//  // Assembly
//  um2MPACTAssemblyNumCells(sp, 0, &nx, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(nx == 1);
//  nx = -1;
//
//  // Lattice
//  um2MPACTLatticeNumCells(sp, 0, &nx, &ny, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(nx == 1);
//  nx = -1;
//  ASSERT(ny == 1);
//  ny = -1;
//
//  // RTMs
//  um2MPACTRTMNumCells(sp, 0, &nx, &ny, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(nx == 1);
//  nx = -1;
//  ASSERT(ny == 1);
//  ny = -1;
//
//  um2DeleteMPACTSpatialPartition(sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  um2Finalize(&ierr);
//  ASSERT(ierr == 0);
//}
//
//TEST_CASE(get_child)
//{
//  Int ierr = -1;
//  um2Initialize("warn", 1, 2, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  void * sp = nullptr;
//  um2ImportMPACTModel("./api_mesh_files/1a.xdmf", &sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  Int id = -1;
//  // Core
//  um2MPACTCoreGetChild(sp, 0, 0, &id, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(id == 0);
//  id = -1;
//  // Assembly
//  um2MPACTAssemblyGetChild(sp, 0, 0, &id, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(id == 0);
//  id = -1;
//  // Lattice
//  um2MPACTLatticeGetChild(sp, 0, 0, 0, &id, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(id == 0);
//  id = -1;
//  // RTM
//  um2MPACTRTMGetChild(sp, 0, 0, 0, &id, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(id == 0);
//  id = -1;
//
//  um2DeleteMPACTSpatialPartition(sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  um2Finalize(&ierr);
//  ASSERT(ierr == 0);
//}
//
//TEST_CASE(coarse_cell_functions)
//{
//  Int ierr = -1;
//  um2Initialize("warn", 1, 2, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  void * sp = nullptr;
//  um2ImportMPACTModel("./api_mesh_files/1a.xdmf", &sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  // numFaces
//  Int n = -1;
//  um2MPACTCoarseCellNumFaces(sp, 0, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 212);
//  n = -1;
//  // dx, dy
//  Float dx = -1;
//  um2MPACTCoarseCellWidth(sp, 0, &dx, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//#if UM2_ENABLE_FLOAT64 == 1
//  Float const expected_dx = 1.26;
//#else
//  Float const expected_dx = 1.26F;
//#endif
//  ASSERT_NEAR(dx, expected_dx, test_eps);
//  dx = -1;
//  um2MPACTRTMWidth(sp, 0, &dx, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT_NEAR(dx, expected_dx, test_eps);
//  dx = -1;
//  Float dy = -1;
//  um2MPACTCoarseCellHeight(sp, 0, &dy, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT_NEAR(dy, expected_dx, test_eps);
//  dy = -1;
//
//  // heights
//  Int * cc_ids = nullptr;
//  Float * cc_heights = nullptr;
//  um2MPACTCoarseCellHeights(sp, &n, &cc_ids, &cc_heights, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 1);
//  n = -1;
//  ASSERT(cc_ids);
//  ASSERT(cc_heights);
//  ASSERT(cc_ids[0] == 0);
//  ASSERT_NEAR(cc_heights[0], static_cast<Float>(2), test_eps);
//  free(cc_ids);
//  cc_ids = nullptr;
//  free(cc_heights);
//  cc_heights = nullptr;
//
//  Float * ass_dzs = nullptr;
//  um2MPACTAssemblyHeights(sp, 0, &n, &ass_dzs, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 1);
//  n = -1;
//  ASSERT_NEAR(ass_dzs[0], static_cast<Float>(2), test_eps);
//
//  // Coarse cell face areas
//  Float * areas = nullptr;
//  um2MPACTCoarseCellFaceAreas(sp, 0, &n, &areas, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 212);
//  n = -1;
//  Float area_sum = 0;
//  for (Int i = 0; i < 212; ++i) {
//    area_sum += areas[i];
//  }
//  ASSERT_NEAR(area_sum, expected_dx * expected_dx, test_eps);
//  free(areas);
//  areas = nullptr;
//
//  // material ids
//  MaterialID * mat_ids = nullptr;
//  um2MPACTCoarseCellMaterialIDs(sp, 0, &mat_ids, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 212);
//  n = -1;
//  ASSERT(mat_ids);
//  ASSERT(mat_ids[0] == 1);
//  ASSERT(mat_ids[1] == 1);
//
//  // find face
//  Int face_id = -2;
//  um2MPACTCoarseCellFaceContaining(sp, 0, static_cast<Float>(0.1),
//                                   static_cast<Float>(0.01), &face_id, &ierr);
//  ASSERT(face_id == 179);
//  face_id = -2;
//  um2MPACTCoarseCellFaceContaining(sp, 0, static_cast<Float>(0.6),
//                                   static_cast<Float>(0.54), &face_id, &ierr);
//  ASSERT(face_id == 0);
//  face_id = -2;
//
//  um2DeleteMPACTSpatialPartition(sp, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  um2Finalize(&ierr);
//  ASSERT(ierr == 0);
//}
//
//TEST_CASE(intersect)
//{
//  Int ierr = -1;
//  um2Initialize("warn", 1, 2, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  void * sp = nullptr;
//  // Cheat with some c++
//  um2::mpact::SpatialPartition model;
//  model.makeCoarseCell({1, 1});
//  model.makeCoarseCell({1, 1});
//  model.makeCoarseCell({1, 1});
//  model.makeRTM({
//      {2, 2},
//      {0, 1}
//  });
//  model.makeLattice({{0}});
//  model.makeAssembly({0});
//  model.makeCore({{0}});
//  model.importCoarseCells("./mpact_mesh_files/coarse_cells.inp");
//  sp = &model;
//
//  Int const buffer_size = 6;
//  Int n = buffer_size;
//  auto * buffer = static_cast<Float *>(malloc(sizeof(Float) * buffer_size));
//  auto ox = static_cast<Float>(0.0);
//  auto oy = static_cast<Float>(0.5);
//  auto dx = static_cast<Float>(1.0);
//  auto dy = static_cast<Float>(0.0);
//  um2MPACTIntersectCoarseCell(sp, 0, ox, oy, dx, dy, buffer, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 4);
//  n = buffer_size;
//  ASSERT_NEAR(buffer[0], static_cast<Float>(0.0), test_eps);
//  ASSERT_NEAR(buffer[1], static_cast<Float>(0.5), test_eps);
//  ASSERT_NEAR(buffer[2], static_cast<Float>(0.5), test_eps);
//  ASSERT_NEAR(buffer[3], static_cast<Float>(1.0), test_eps);
//
//  um2MPACTIntersectCoarseCell(sp, 1, ox, oy, dx, dy, buffer, &n, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(n == 4);
//  n = buffer_size;
//  ASSERT_NEAR(buffer[0], static_cast<Float>(0.0), test_eps);
//  ASSERT_NEAR(buffer[1], static_cast<Float>(0.5), test_eps);
//  ASSERT_NEAR(buffer[2], static_cast<Float>(0.5), test_eps);
//  ASSERT_NEAR(buffer[3], static_cast<Float>(1.0), test_eps);
//  free(buffer);
//
//  Int mesh_type = -1;
//  Int nverts = -1;
//  Int nfaces = -1;
//  Int * fv = nullptr;
//  Float * vertices = nullptr;
//  um2MPACTCoarseCellFaceData(sp, 0, &mesh_type, &nverts, &nfaces, &vertices, &fv, &ierr);
//  ASSERT(ierr == 0);
//  ierr = -1;
//  ASSERT(mesh_type == 3);
//  mesh_type = -1;
//  ASSERT(nverts == 4);
//  nverts = -1;
//  ASSERT(nfaces == 2);
//  nfaces = -1;
//  ASSERT(fv);
//  ASSERT(fv[0] == 0);
//  fv[0] = -1;
//  ASSERT(fv[1] == 1);
//  fv[1] = -1;
//  ASSERT(fv[2] == 2);
//  fv[2] = -1;
//  ASSERT(fv[3] == 2);
//  fv[3] = -1;
//  ASSERT(fv[4] == 3);
//  fv[4] = -1;
//  ASSERT(fv[5] == 0);
//  fv[5] = -1;
//  ASSERT(vertices);
//  auto const zero = static_cast<Float>(0);
//  auto const one = static_cast<Float>(1);
//  ASSERT_NEAR(vertices[0], zero, test_eps);
//  ASSERT_NEAR(vertices[1], zero, test_eps);
//  ASSERT_NEAR(vertices[2], one, test_eps);
//  ASSERT_NEAR(vertices[3], zero, test_eps);
//  ASSERT_NEAR(vertices[4], one, test_eps);
//  ASSERT_NEAR(vertices[5], one, test_eps);
//  ASSERT_NEAR(vertices[6], zero, test_eps);
//  ASSERT_NEAR(vertices[7], one, test_eps);
//
//  um2Finalize(&ierr);
//  ASSERT(ierr == 0);
//}

TEST_SUITE(c_api)
{
  TEST(data_sizes);
  TEST(malloc_free);
  TEST(initialize_finalize);
//  TEST(new_delete_spatial_partition);
//  TEST(import_MPACT_model);
//  TEST(mpact_num_cells);
//  TEST(get_child);
//  TEST(coarse_cell_functions);
//  TEST(intersect);
}

auto
main() -> int
{
  RUN_SUITE(c_api);
  return 0;
}
