#include <um2c.h>
#include <um2/stdlib/algorithm/is_sorted.hpp>

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

TEST_CASE(new_delete_mpact_model)
{
  um2Initialize();
  void * sp = nullptr;
  um2NewMPACTModel(&sp);
  ASSERT(sp != nullptr);
  um2DeleteMPACTModel(sp);
  um2Finalize();
}

TEST_CASE(read_mpact_model)
{
  um2Initialize();
  void * sp = nullptr;
  um2ReadMPACTModel("./api_mesh_files/1a_nogap.xdmf", &sp);
  ASSERT(sp != nullptr);

  auto const & model = *reinterpret_cast<um2::mpact::Model *>(sp);

  // Coarse cells
  ASSERT(model.numCoarseCells() == 1);
  Int n = -1;
  um2MPACTNumCoarseCells(sp, &n);
  ASSERT(n == 1);
  n = -1;

  // RTMs
  ASSERT(model.numRTMs() == 1);
  um2MPACTNumRTMs(sp, &n);
  ASSERT(n == 1);
  n = -1;

  // Lattices
  ASSERT(model.numLattices() == 1);
  um2MPACTNumLattices(sp, &n);
  ASSERT(n == 1);
  n = -1;

  // Assemblies
  ASSERT(model.numAssemblies() == 1);
  um2MPACTNumAssemblies(sp, &n);
  ASSERT(n == 1);
  n = -1;

  um2DeleteMPACTModel(sp);
  um2Finalize();
}

TEST_CASE(mpact_num_cells)
{
  um2Initialize();
  void * sp = nullptr;
  um2ReadMPACTModel("./api_mesh_files/1a_nogap.xdmf", &sp);
  Int nx = -1;
  Int ny = -1;

  // Core
  um2MPACTCoreNumCells(sp, &nx, &ny);
  ASSERT(nx == 1);
  nx = -1;
  ASSERT(ny == 1);
  ny = -1;

  // Assembly
  um2MPACTAssemblyNumCells(sp, 0, &nx);
  ASSERT(nx == 1);
  nx = -1;

  // Lattice
  um2MPACTLatticeNumCells(sp, 0, &nx, &ny);
  ASSERT(nx == 1);
  nx = -1;
  ASSERT(ny == 1);
  ny = -1;

  // RTMs
  um2MPACTRTMNumCells(sp, 0, &nx, &ny);
  ASSERT(nx == 1);
  nx = -1;
  ASSERT(ny == 1);
  ny = -1;

  um2DeleteMPACTModel(sp);
  um2Finalize();
}

TEST_CASE(mpact_get_child)
{
  um2Initialize();
  void * sp = nullptr;
  um2ReadMPACTModel("./api_mesh_files/1a_nogap.xdmf", &sp);
  Int id = -1;
  
  // Core
  um2MPACTCoreGetChild(sp, 0, 0, &id);
  ASSERT(id == 0);
  id = -1;

  // Assembly
  um2MPACTAssemblyGetChild(sp, 0, 0, &id);
  ASSERT(id == 0);
  id = -1;

  // Lattice
  um2MPACTLatticeGetChild(sp, 0, 0, 0, &id);
  ASSERT(id == 0);
  id = -1;

  // RTM
  um2MPACTRTMGetChild(sp, 0, 0, 0, &id);
  ASSERT(id == 0);
  id = -1;

  um2DeleteMPACTModel(sp);
  um2Finalize();
}

TEST_CASE(coarse_cell_functions)
{
  um2Initialize();
  void * sp = nullptr;
  um2ReadMPACTModel("./api_mesh_files/1a_nogap.xdmf", &sp);
  ASSERT(sp != nullptr);

  // numFaces
  Int n = -1;
  um2MPACTCoarseCellNumFaces(sp, 0, &n);
  ASSERT(n == 48); 
  n = -1;

  // width, height 
  Float dx = -1;
  auto const expected_dx = castIfNot<Float>(1.26);
  auto const test_eps = um2::eps_distance; 
  um2MPACTCoarseCellWidth(sp, 0, &dx);
  ASSERT_NEAR(dx, expected_dx, test_eps);

  um2MPACTRTMWidth(sp, 0, &dx);
  ASSERT_NEAR(dx, expected_dx, test_eps);
  dx = -1;

  Float dy = -1;
  um2MPACTCoarseCellHeight(sp, 0, &dy);
  ASSERT_NEAR(dy, expected_dx, test_eps);

  // heights
  Int * cc_ids = nullptr;
  Float * cc_heights = nullptr;
  um2MPACTCoarseCellHeights(sp, &n, &cc_ids, &cc_heights);
  ASSERT(n == 1);
  n = -1;
  ASSERT(cc_ids);
  ASSERT(cc_heights);
  ASSERT(cc_ids[0] == 0);
  ASSERT_NEAR(cc_heights[0], 1, test_eps);
  free(cc_ids);
  cc_ids = nullptr;
  free(cc_heights);
  cc_heights = nullptr;

  Float ass_dzs[10];
  for (auto & dz : ass_dzs) {
    dz = 0;
  }
  um2MPACTAssemblyHeights(sp, 0, ass_dzs);
  ASSERT_NEAR(ass_dzs[0], 1, test_eps);
  for (Int i = 1; i < 10; ++i) {
    ASSERT_NEAR(ass_dzs[i], 0, test_eps);
  }

  // Coarse cell face areas
  Float areas[48]; 
  um2MPACTCoarseCellFaceAreas(sp, 0, areas);
  for (Int i = 0; i < 24; ++i) {
    ASSERT_NEAR(areas[i], castIfNot<Float>(0.02196132438887047), test_eps); 
  }
  Float area_sum = 0;
  for (auto area : areas) {
    area_sum += area;
  }
  ASSERT_NEAR(area_sum, expected_dx * expected_dx, test_eps);

  // material ids
  MatID mat_ids[48];
  um2MPACTCoarseCellMaterialIDs(sp, 0, mat_ids);
  ASSERT(mat_ids[0] == 0);
  ASSERT(mat_ids[24] == 1);
  ASSERT(mat_ids[44] == 2);

  // find face
  auto x = castIfNot<Float>(0.7);
  auto const y = castIfNot<Float>(0.64);
  Int face_id = -2;
  um2MPACTCoarseCellFaceContaining(sp, 0, x, y, &face_id); 
  ASSERT(face_id == 0); 
  face_id = -2;

  x = castIfNot<Float>(0.9);
  um2MPACTCoarseCellFaceContaining(sp, 0, x, y, &face_id); 
  ASSERT(face_id == 8);
  face_id = -2;

  x = castIfNot<Float>(1.0);
  um2MPACTCoarseCellFaceContaining(sp, 0, x, y, &face_id); 
  ASSERT(face_id == 16);
  face_id = -2;
  
  x = castIfNot<Float>(1.0723);
  um2MPACTCoarseCellFaceContaining(sp, 0, x, y, &face_id); 
  ASSERT(face_id == 24);
  face_id = -2;
  
  x = castIfNot<Float>(1.155);
  um2MPACTCoarseCellFaceContaining(sp, 0, x, y, &face_id); 
  ASSERT(face_id == 32);
  face_id = -2;
  
  x = castIfNot<Float>(1.2325);
  um2MPACTCoarseCellFaceContaining(sp, 0, x, y, &face_id); 
  ASSERT(face_id == 40);
  face_id = -2;

  // Intersect
  Float buffer[256];
  n = -1;
  auto const ox = castIfNot<Float>(0.0);
  auto const oy = castIfNot<Float>(0.62);
  dx = castIfNot<Float>(1.0);
  dy = castIfNot<Float>(0.0);
  um2MPACTIntersectCoarseCell(sp, 0, ox, oy, dx, dy, buffer, &n);
  ASSERT(n == 28);
  for (Int i = 0; i < n; ++i) {
    ASSERT(buffer[i] >= 0);
  }
  ASSERT(um2::is_sorted(buffer, buffer + n));

  // FaceData
  Int mesh_type = -1;
  Int num_vertices = -1;
  Int num_faces = -1;
  Float * vertices = nullptr;
  Int * faces = nullptr;
  um2MPACTCoarseCellFaceData(sp, 0, &mesh_type, &num_vertices, &num_faces,
      &vertices, &faces);
  ASSERT(mesh_type == 8);
  ASSERT(num_faces == 48);
  ASSERT(vertices != nullptr);
  ASSERT(faces != nullptr);

  um2Free(vertices);
  um2Free(faces);
  um2DeleteMPACTModel(sp);
  um2Finalize();
}

TEST_SUITE(c_api)
{
  TEST(data_sizes);
  TEST(malloc_free);
  TEST(initialize_finalize);
  TEST(new_delete_mpact_model);
  TEST(read_mpact_model);
  TEST(mpact_num_cells);
  TEST(mpact_get_child);
  TEST(coarse_cell_functions);
}

auto
main() -> int
{
  RUN_SUITE(c_api);
  return 0;
}
