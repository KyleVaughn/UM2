#include <um2/mpact/SpatialPartition.hpp>
#include <um2/mpact/io.hpp>

#include <iostream>

#include "../test_macros.hpp"

#include <fstream>

#if UM2_ENABLE_FLOAT64 == 1
constexpr Float test_eps = 1e-6;
#else
constexpr auto test_eps = static_cast<Float>(1e-6);
#endif

// template <typename T, typename I>
// TEST_CASE(test_make_cylindrical_pin_mesh)
// um2::mpact::SpatialPartition model;
// std::vector<double> const radii = {0.4096, 0.475, 0.575};
// double const pitch = 1.26;
// std::vector<int> const num_rings = {3, 1, 1};
// int const na = 8;
//
// int id = -1;
// id = model.make_cylindrical_pin_mesh(radii, pitch, num_rings, na, 1);
// ASSERT(id == 0, "id should be 0");
// ASSERT(model.quad.size() == 1, "size");
// int total_rings = std::reduce(num_rings.begin(), num_rings.end());
// int nfaces = (total_rings + 1) * na;
// ASSERT(num_faces(model.quad[0]) == nfaces, "num_faces");
// auto aabb = um2::bounding_box(model.quad[0]);
// ASSERT_NEAR(um2::width(aabb), 1.26, 1e-6, "width");
// ASSERT_NEAR(um2::height(aabb), 1.26, 1e-6, "height");
// ASSERT_NEAR(aabb.minima[0], 0, 1e-6, "minima[0]");
// ASSERT_NEAR(aabb.minima[1], 0, 1e-6, "minima[1]");
//
// id = model.make_cylindrical_pin_mesh(radii, pitch, num_rings, na, 2);
// ASSERT(id == 0, "id should be 0");
// ASSERT(model.quadratic_quad.size() == 1, "size");
// ASSERT(num_faces(model.quadratic_quad[0]) == nfaces, "num_faces");
// aabb = um2::bounding_box(model.quadratic_quad[0]);
// ASSERT_NEAR(um2::width(aabb), 1.26, 1e-6, "width");
// ASSERT_NEAR(um2::height(aabb), 1.26, 1e-6, "height");
// ASSERT_NEAR(aabb.minima[0], 0, 1e-6, "minima[0]");
// ASSERT_NEAR(aabb.minima[1], 0, 1e-6, "minima[1]");
//
// std::vector<um2::Material> materials;
// um2::Vector<int8_t> material_ids;
// um2::Material uo2("UO2", "forestgreen");
// um2::Material clad("Clad", "lightgray");
// um2::Material water("Water", "lightblue");
// materials.insert(materials.end(), 3 * 8, uo2);
// materials.insert(materials.end(), 1 * 8, clad);
// materials.insert(materials.end(), 2 * 8, water);
// material_ids.insert(material_ids.end(), 3 * 8, 0);
// material_ids.insert(material_ids.end(), 1 * 8, 1);
// material_ids.insert(material_ids.end(), 2 * 8, 2);
// id = model.makeCoarseCell(2, 0, materials);
// ASSERT(id == 0, "id should be 0");
// ASSERT_NEAR(model.coarse_cells[0].dxdy[0], pitch, 1e-6, "dxdy");
// ASSERT_NEAR(model.coarse_cells[0].dxdy[1], pitch, 1e-6, "dxdy");
// ASSERT(model.coarse_cells[0].mesh_type == 2, "mesh_type");
// ASSERT(model.coarse_cells[0].mesh_id == 0, "mesh_id");
// ASSERT(model.coarse_cells[0].material_ids == material_ids, "material_ids");
// END_TEST_CASE

// template <typename T, typename I>
// TEST_CASE(test_make_rectangular_pin_mesh)
//{
// um2::mpact::SpatialPartition model;
// um2::Vec2<Float> dxdy(2, 1);
//
// int id = -1;
// int nx = 1;
// int ny = 1;
// id = model.make_rectangular_pin_mesh(dxdy, nx, ny);
// ASSERT(id == 0, "id should be 0");
// ASSERT(model.quad.size() == 1, "size");
// ASSERT(num_faces(model.quad[0]) == nx * ny, "num_faces");
// auto aabb = um2::bounding_box(model.quad[0]);
// ASSERT_NEAR(um2::width(aabb), dxdy[0], 1e-6, "width");
// ASSERT_NEAR(um2::height(aabb), dxdy[1], 1e-6, "height");
// ASSERT_NEAR(aabb.minima[0], 0, 1e-6, "minima[0]");
// ASSERT_NEAR(aabb.minima[1], 0, 1e-6, "minima[1]");
//
// std::vector<um2::Material>
// materials(nx * ny, um2::Material("A", "red"));
// um2::Vector<int8_t>
// material_ids(nx * ny, 0);
// id = model.makeCoarseCell(2, 0, materials);
// ASSERT(id == 0, "id should be 0");
// ASSERT_NEAR(model.coarse_cells[0].dxdy[0], dxdy[0], 1e-6, "dxdy");
// ASSERT_NEAR(model.coarse_cells[0].dxdy[1], dxdy[1], 1e-6, "dxdy");
// ASSERT(model.coarse_cells[0].mesh_type == 2, "mesh_type");
// ASSERT(model.coarse_cells[0].mesh_id == 0, "mesh_id");
// ASSERT(model.coarse_cells[0].material_ids == material_ids, "material_ids");
// }

TEST_CASE(makeCoarseCell)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<Float> const dxdy(2, 1);
  Size const id = model.makeCoarseCell(dxdy);
  ASSERT(id == 0);
  ASSERT(model.numCoarseCells() == 1);
  ASSERT(um2::isApprox(model.coarse_cells[0].dxdy, dxdy));
  ASSERT(model.coarse_cells[0].mesh_type == um2::MeshType::None);
  ASSERT(model.coarse_cells[0].mesh_id == -1);
  ASSERT(model.coarse_cells[0].material_ids.empty());
  Size const id2 = model.makeCoarseCell(dxdy);
  ASSERT(id2 == 1);
  ASSERT(model.numCoarseCells() == 2);
  ASSERT(um2::isApprox(model.coarse_cells[1].dxdy, dxdy));
  ASSERT(model.coarse_cells[1].mesh_type == um2::MeshType::None);
  ASSERT(model.coarse_cells[1].mesh_id == -1);
  ASSERT(model.coarse_cells[1].material_ids.empty());
}

TEST_CASE(makeRTM)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<Float> const dxdy(2, 1);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 0);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 1);
  ASSERT(model.numCoarseCells() == 2);
  std::vector<std::vector<Size>> const cc_ids = {
      {0, 1}
  };
  Size id = model.makeRTM(cc_ids);
  ASSERT(id == 0);
  ASSERT(model.numRTMs() == 1);
  ASSERT(model.rtms[0].children.size() == 2);
  ASSERT(model.rtms[0].children[0] == 0);
  ASSERT(model.rtms[0].children[1] == 1);
  um2::RectilinearGrid2<Float> const & grid = model.rtms[0].grid;
  ASSERT(grid.divs[0].size() == 3);
  ASSERT(grid.divs[1].size() == 2);
  ASSERT_NEAR(grid.divs[0][0], 0, test_eps);
  ASSERT_NEAR(grid.divs[0][1], 2, test_eps);
  ASSERT_NEAR(grid.divs[0][2], 4, test_eps);
  ASSERT_NEAR(grid.divs[1][0], 0, test_eps);
  ASSERT_NEAR(grid.divs[1][1], 1, test_eps);
  model.clear();

  std::vector<std::vector<Size>> const cc_ids2 = {
      {2, 3},
      {0, 1}
  };
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 0);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 1);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 2);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 3);
  id = model.makeRTM(cc_ids2);
  ASSERT(id == 0);
  ASSERT(model.rtms.size() == 1);
  ASSERT(model.rtms[0].children.size() == 4);
  ASSERT(model.rtms[0].children[0] == 0);
  ASSERT(model.rtms[0].children[1] == 1);
  ASSERT(model.rtms[0].children[2] == 2);
  ASSERT(model.rtms[0].children[3] == 3);
  um2::RectilinearGrid2<Float> const & grid2 = model.rtms[0].grid;
  ASSERT(grid2.divs[0].size() == 3);
  ASSERT(grid2.divs[1].size() == 3);
  ASSERT_NEAR(grid2.divs[0][0], 0, test_eps);
  ASSERT_NEAR(grid2.divs[0][1], 2, test_eps);
  ASSERT_NEAR(grid2.divs[0][2], 4, test_eps);
  ASSERT_NEAR(grid2.divs[1][0], 0, test_eps);
  ASSERT_NEAR(grid2.divs[1][1], 1, test_eps);
  ASSERT_NEAR(grid2.divs[1][2], 2, test_eps);
}

TEST_CASE(makeLattice)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<Float> const dxdy0(3, 3);
  um2::Vec2<Float> const dxdy1(4, 4);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy0) == 0);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy1) == 1);
  std::vector<std::vector<Size>> const cc_ids_44 = {
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      {0, 0, 0, 0}
  };
  std::vector<std::vector<Size>> const cc_ids_33 = {
      {1, 1, 1},
      {1, 1, 1},
      {1, 1, 1}
  };
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeRTM(cc_ids_33) == 0);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeRTM(cc_ids_44) == 1);
  std::vector<std::vector<Size>> const rtm_ids = {
      {0, 1}
  };
  Size const id = model.makeLattice(rtm_ids);
  ASSERT(id == 0);
  ASSERT(model.lattices.size() == 1);
  ASSERT(model.lattices[0].children.size() == 2);
  ASSERT(model.lattices[0].children[0] == 0);
  ASSERT(model.lattices[0].children[1] == 1);
  um2::RegularGrid2<Float> const & grid = model.lattices[0].grid;
  ASSERT(grid.numXCells() == 2);
  ASSERT(grid.numYCells() == 1);
  ASSERT_NEAR(grid.spacing[0], 12, test_eps);
  ASSERT_NEAR(grid.spacing[1], 12, test_eps);
  ASSERT_NEAR(grid.minima[0], 0, test_eps);
  ASSERT_NEAR(grid.minima[1], 0, test_eps);
}

TEST_CASE(makeAssembly)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<Float> const dxdy(1, 1);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 0);
  std::vector<std::vector<Size>> const cc_ids = {
      {0, 0},
      {0, 0}
  };
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeRTM(cc_ids) == 0);
  std::vector<std::vector<Size>> const rtm_ids = {{0}};
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeLattice(rtm_ids) == 0);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeLattice(rtm_ids) == 1);
  std::vector<Size> const lat_ids = {0, 1, 0};
  std::vector<Float> const lat_z = {0, 2, 3, 4};
  Size const id = model.makeAssembly(lat_ids, lat_z);
  ASSERT(id == 0);
  ASSERT(model.assemblies.size() == 1);
  ASSERT(model.assemblies[0].children.size() == 3);
  ASSERT(model.assemblies[0].children[0] == 0);
  ASSERT(model.assemblies[0].children[1] == 1);
  ASSERT(model.assemblies[0].children[2] == 0);
  um2::RectilinearGrid1<Float> const & grid = model.assemblies[0].grid;
  ASSERT(grid.divs[0].size() == 4);
  ASSERT_NEAR(grid.divs[0][0], 0, test_eps);
  ASSERT_NEAR(grid.divs[0][1], 2, test_eps);
  ASSERT_NEAR(grid.divs[0][2], 3, test_eps);
  ASSERT_NEAR(grid.divs[0][3], 4, test_eps);
}

TEST_CASE(makeAssembly_2d)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<Float> const dxdy(1, 1);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 0);
  std::vector<std::vector<Size>> const cc_ids = {
      {0, 0},
      {0, 0}
  };
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeRTM(cc_ids) == 0);
  std::vector<std::vector<Size>> const rtm_ids = {{0}};
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeLattice(rtm_ids) == 0);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeLattice(rtm_ids) == 1);
  std::vector<Size> const lat_ids = {0};
  Size const id = model.makeAssembly(lat_ids);
  ASSERT(id == 0);
  ASSERT(model.assemblies.size() == 1);
  ASSERT(model.assemblies[0].children.size() == 1);
  ASSERT(model.assemblies[0].children[0] == 0);
  um2::RectilinearGrid1<Float> const & grid = model.assemblies[0].grid;
  ASSERT(grid.divs[0].size() == 2);
  ASSERT_NEAR(grid.divs[0][0], -1, test_eps);
  ASSERT_NEAR(grid.divs[0][1], 1, test_eps);
}

TEST_CASE(makeCore)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<Float> const dxdy(2, 1);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeCoarseCell(dxdy) == 0);

  std::vector<std::vector<Size>> const cc_ids = {{0}};
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeRTM(cc_ids) == 0);

  std::vector<std::vector<Size>> const rtm_ids = {{0}};
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeLattice(rtm_ids) == 0);

  std::vector<Size> const lat_ids1 = {0, 0, 0};
  std::vector<Float> const lat_z1 = {0, 2, 3, 4};
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeAssembly(lat_ids1, lat_z1) == 0);
  std::vector<Size> const lat_ids2 = {0, 0};
  std::vector<Float> const lat_z2 = {0, 3, 4};
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeAssembly(lat_ids2, lat_z2) == 1);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeAssembly(lat_ids1, lat_z1) == 2);
  // cppcheck-suppress assertWithSideEffect; justified
  ASSERT(model.makeAssembly(lat_ids2, lat_z2) == 3);

  std::vector<std::vector<Size>> const asy_ids = {
      {2, 3},
      {0, 1}
  };
  Size const id = model.makeCore(asy_ids);
  ASSERT(id == 0);
  ASSERT(model.core.children.size() == 4);
  ASSERT(model.core.children[0] == 0);
  ASSERT(model.core.children[1] == 1);
  ASSERT(model.core.children[2] == 2);
  ASSERT(model.core.children[3] == 3);
  ASSERT(model.core.grid.divs[0].size() == 3);
  ASSERT(model.core.grid.divs[1].size() == 3);
  ASSERT_NEAR(model.core.grid.divs[0][0], 0, test_eps);
  ASSERT_NEAR(model.core.grid.divs[0][1], 2, test_eps);
  ASSERT_NEAR(model.core.grid.divs[0][2], 4, test_eps);
  ASSERT_NEAR(model.core.grid.divs[1][0], 0, test_eps);
  ASSERT_NEAR(model.core.grid.divs[1][1], 1, test_eps);
  ASSERT_NEAR(model.core.grid.divs[1][2], 2, test_eps);
}

TEST_CASE(importCoarseCells)
{
  using CoarseCell = um2::mpact::SpatialPartition::CoarseCell;
  um2::mpact::SpatialPartition model;
  model.makeCoarseCell({1, 1});
  model.makeCoarseCell({1, 1});
  model.makeCoarseCell({1, 1});
  model.makeRTM({
      {2, 2},
      {0, 1}
  });
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});
  model.importCoarseCells("./mpact_mesh_files/coarse_cells.inp");

  ASSERT(model.numAssemblies() == 1);
  ASSERT(model.numLattices() == 1);
  ASSERT(model.numRTMs() == 1);
  ASSERT(model.numCoarseCells() == 3);

  ASSERT(model.tri.size() == 2);
  CoarseCell const & cell = model.coarse_cells[0];
  ASSERT(cell.mesh_type == um2::MeshType::Tri);
  ASSERT(cell.mesh_id == 0);
  ASSERT(cell.material_ids.size() == 2);
  ASSERT(cell.material_ids[0] == 1);
  ASSERT(cell.material_ids[1] == 2);
  um2::TriMesh<2, Float, Int> const & tri_mesh = model.tri[0];
  ASSERT(tri_mesh.numVertices() == 4);
  ASSERT(um2::isApprox(tri_mesh.vertices[0], {0, 0}));
  ASSERT(um2::isApprox(tri_mesh.vertices[1], {1, 0}));
  ASSERT(um2::isApprox(tri_mesh.vertices[2], {1, 1}));
  ASSERT(um2::isApprox(tri_mesh.vertices[3], {0, 1}));
  ASSERT(tri_mesh.fv[0][0] == 0);
  ASSERT(tri_mesh.fv[0][1] == 1);
  ASSERT(tri_mesh.fv[0][2] == 2);
  ASSERT(tri_mesh.fv[1][0] == 2);
  ASSERT(tri_mesh.fv[1][1] == 3);
  ASSERT(tri_mesh.fv[1][2] == 0);

  CoarseCell const & cell1 = model.coarse_cells[1];
  ASSERT(cell1.mesh_type == um2::MeshType::Tri);
  ASSERT(cell1.mesh_id == 1);
  ASSERT(cell1.material_ids.size() == 2);
  ASSERT(cell1.material_ids[0] == 1);
  ASSERT(cell1.material_ids[1] == 0);
  um2::TriMesh<2, Float, Int> const & tri_mesh1 = model.tri[1];
  ASSERT(tri_mesh1.vertices.size() == 4);
  ASSERT(um2::isApprox(tri_mesh1.vertices[0], {0, 0}));
  ASSERT(um2::isApprox(tri_mesh1.vertices[1], {0, 1}));
  ASSERT(um2::isApprox(tri_mesh1.vertices[2], {1, 0}));
  ASSERT(um2::isApprox(tri_mesh1.vertices[3], {1, 1}));
  ASSERT(tri_mesh1.fv[0][0] == 0);
  ASSERT(tri_mesh1.fv[0][1] == 2);
  ASSERT(tri_mesh1.fv[0][2] == 1);
  ASSERT(tri_mesh1.fv[1][0] == 2);
  ASSERT(tri_mesh1.fv[1][1] == 3);
  ASSERT(tri_mesh1.fv[1][2] == 1);

  CoarseCell const & cell2 = model.coarse_cells[2];
  ASSERT(cell2.mesh_type == um2::MeshType::Quad);
  ASSERT(cell2.mesh_id == 0);
  ASSERT(cell2.material_ids.size() == 1);
  ASSERT(cell2.material_ids[0] == 0);
  um2::QuadMesh<2, Float, Int> const & quad_mesh = model.quad[0];
  ASSERT(quad_mesh.vertices.size() == 4);
  ASSERT(um2::isApprox(quad_mesh.vertices[0], {1, 0}));
  ASSERT(um2::isApprox(quad_mesh.vertices[1], {0, 0}));
  ASSERT(um2::isApprox(quad_mesh.vertices[2], {1, 1}));
  ASSERT(um2::isApprox(quad_mesh.vertices[3], {0, 1}));
  ASSERT(quad_mesh.fv.size() == 1);
  ASSERT(quad_mesh.fv[0][0] == 1);
  ASSERT(quad_mesh.fv[0][1] == 0);
  ASSERT(quad_mesh.fv[0][2] == 2);
  ASSERT(quad_mesh.fv[0][3] == 3);
}

TEST_CASE(io)
{
  using CoarseCell = um2::mpact::SpatialPartition::CoarseCell;
  um2::mpact::SpatialPartition model_out;
  model_out.makeCoarseCell({1, 1});
  model_out.makeCoarseCell({1, 1});
  model_out.makeCoarseCell({1, 1});
  model_out.makeRTM({
      {2, 2},
      {0, 1}
  });
  model_out.makeLattice({{0}});
  model_out.makeAssembly({0});
  model_out.makeCore({{0}});
  model_out.importCoarseCells("./mpact_mesh_files/coarse_cells.inp");
  std::string const filepath = "./mpact_export_test_model.xdmf";
  um2::exportMesh(filepath, model_out);
  um2::mpact::SpatialPartition model;
  um2::importMesh(filepath, model);

  ASSERT(model.numAssemblies() == 1);
  ASSERT(model.numLattices() == 1);
  ASSERT(model.numRTMs() == 1);
  ASSERT(model.numCoarseCells() == 3);

  ASSERT(model.tri.size() == 2);
  CoarseCell const & cell = model.coarse_cells[0];
  ASSERT(cell.mesh_type == um2::MeshType::Tri);
  ASSERT(cell.mesh_id == 0);
  ASSERT(cell.material_ids.size() == 2);
  ASSERT(cell.material_ids[0] == 1);
  ASSERT(cell.material_ids[1] == 2);
  um2::TriMesh<2, Float, Int> const & tri_mesh = model.tri[0];
  ASSERT(tri_mesh.numVertices() == 4);
  ASSERT(um2::isApprox(tri_mesh.vertices[0], {0, 0}));
  ASSERT(um2::isApprox(tri_mesh.vertices[1], {1, 0}));
  ASSERT(um2::isApprox(tri_mesh.vertices[2], {1, 1}));
  ASSERT(um2::isApprox(tri_mesh.vertices[3], {0, 1}));
  ASSERT(tri_mesh.fv[0][0] == 0);
  ASSERT(tri_mesh.fv[0][1] == 1);
  ASSERT(tri_mesh.fv[0][2] == 2);
  ASSERT(tri_mesh.fv[1][0] == 2);
  ASSERT(tri_mesh.fv[1][1] == 3);
  ASSERT(tri_mesh.fv[1][2] == 0);

  CoarseCell const & cell1 = model.coarse_cells[1];
  ASSERT(cell1.mesh_type == um2::MeshType::Tri);
  ASSERT(cell1.mesh_id == 1);
  ASSERT(cell1.material_ids.size() == 2);
  ASSERT(cell1.material_ids[0] == 1);
  ASSERT(cell1.material_ids[1] == 0);
  um2::TriMesh<2, Float, Int> const & tri_mesh1 = model.tri[1];
  ASSERT(tri_mesh1.vertices.size() == 4);
  ASSERT(um2::isApprox(tri_mesh1.vertices[0], {0, 0}));
  ASSERT(um2::isApprox(tri_mesh1.vertices[1], {0, 1}));
  ASSERT(um2::isApprox(tri_mesh1.vertices[2], {1, 0}));
  ASSERT(um2::isApprox(tri_mesh1.vertices[3], {1, 1}));
  ASSERT(tri_mesh1.fv[0][0] == 0);
  ASSERT(tri_mesh1.fv[0][1] == 2);
  ASSERT(tri_mesh1.fv[0][2] == 1);
  ASSERT(tri_mesh1.fv[1][0] == 2);
  ASSERT(tri_mesh1.fv[1][1] == 3);
  ASSERT(tri_mesh1.fv[1][2] == 1);

  CoarseCell const & cell2 = model.coarse_cells[2];
  ASSERT(cell2.mesh_type == um2::MeshType::Quad);
  ASSERT(cell2.mesh_id == 0);
  ASSERT(cell2.material_ids.size() == 1);
  ASSERT(cell2.material_ids[0] == 0);
  um2::QuadMesh<2, Float, Int> const & quad_mesh = model.quad[0];
  ASSERT(quad_mesh.vertices.size() == 4);
  ASSERT(um2::isApprox(quad_mesh.vertices[0], {1, 0}));
  ASSERT(um2::isApprox(quad_mesh.vertices[1], {0, 0}));
  ASSERT(um2::isApprox(quad_mesh.vertices[2], {1, 1}));
  ASSERT(um2::isApprox(quad_mesh.vertices[3], {0, 1}));
  ASSERT(quad_mesh.fv.size() == 1);
  ASSERT(quad_mesh.fv[0][0] == 1);
  ASSERT(quad_mesh.fv[0][1] == 0);
  ASSERT(quad_mesh.fv[0][2] == 2);
  ASSERT(quad_mesh.fv[0][3] == 3);

  int stat = std::remove("./mpact_export_test_model.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./mpact_export_test_model.h5");
  ASSERT(stat == 0);
}
//// template <typename T, typename I>
//// TEST_CASE(test_coarse_cell_face_areas)
//// um2::mpact::SpatialPartition model;
//// model.makeCoarseCell({1, 1});
//// model.makeCoarseCell({1, 1});
//// model.makeCoarseCell({1, 1});
//// model.make_rtm({
////     {2, 2},
////     {0, 1}
//// });
//// model.make_lattice({{0}});
//// model.make_assembly({0});
//// model.make_core({{0}});
//// model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
////
//// um2::Vector<Float> areas;
//// model.coarse_cell_face_areas(0, areas);
//// ASSERT(areas.size() == 2, "areas");
//// ASSERT_NEAR(areas[0], 0.5, 1e-4, "areas");
//// ASSERT_NEAR(areas[1], 0.5, 1e-4, "areas");
//// model.coarse_cell_face_areas(1, areas);
//// ASSERT(areas.size() == 2, "areas");
//// ASSERT_NEAR(areas[0], 0.5, 1e-4, "areas");
//// ASSERT_NEAR(areas[1], 0.5, 1e-4, "areas");
//// model.coarse_cell_face_areas(2, areas);
//// ASSERT(areas.size() == 1, "areas");
//// ASSERT_NEAR(areas[0], 1.0, 1e-4, "areas");
//// END_TEST_CASE
////
//// template <typename T, typename I>
//// TEST_CASE(test_coarse_cell_find_face)
//// um2::mpact::SpatialPartition model;
//// model.makeCoarseCell({1, 1});
//// model.makeCoarseCell({1, 1});
//// model.makeCoarseCell({1, 1});
//// model.make_rtm({
////     {2, 2},
////     {0, 1}
//// });
//// model.make_lattice({{0}});
//// model.make_assembly({0});
//// model.make_core({{0}});
//// model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
////
//// length_t face_id = model.coarse_cell_find_face(
////     2, um2::Point2<Float>(static_cast<Float>(0.5), static_cast<Float>(0.5)));
//// ASSERT(face_id == 0, "face_id");
//// face_id = -2;
//// face_id = model.coarse_cell_find_face(
////     2, um2::Point2<Float>(static_cast<Float>(0.5), static_cast<Float>(1.5)));
//// ASSERT(face_id == -1, "face_id");
//// face_id = -2;
////
//// face_id = model.coarse_cell_find_face(
////     1, um2::Point2<Float>(static_cast<Float>(0.5), static_cast<Float>(0.05)));
//// ASSERT(face_id == 0, "face_id");
//// face_id = -2;
//// face_id = model.coarse_cell_find_face(
////     1, um2::Point2<Float>(static_cast<Float>(0.5), static_cast<Float>(-0.05)));
//// ASSERT(face_id == -1, "face_id");
//// face_id = -2;
//// face_id = model.coarse_cell_find_face(
////     1, um2::Point2<Float>(static_cast<Float>(0.5), static_cast<Float>(0.95)));
//// ASSERT(face_id == 1, "face_id");
//// face_id = -2;
//// END_TEST_CASE
////
//// template <typename T, typename I>
//// TEST_CASE(test_coarse_cell_ray_intersect)
//// um2::mpact::SpatialPartition model;
//// model.makeCoarseCell({1, 1});
//// model.makeCoarseCell({1, 1});
//// model.makeCoarseCell({1, 1});
//// model.make_rtm({
////     {2, 2},
////     {0, 1}
//// });
//// model.make_lattice({{0}});
//// model.make_assembly({0});
//// model.make_core({{0}});
//// model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
////
//// um2::Ray2<Float> ray(um2::Point2<Float>(static_cast<Float>(0),
/// static_cast<Float>(0.5)), /                  um2::Vec2<Float>(1, 0)); / int n = 8; / T
///* intersections = new T[n]; / model.intersect_coarse_cell(0, ray, intersections, &n);
//// ASSERT(n == 4, "intersections");
//// for (int i = 0; i < n; i++)
////   std::cout << intersections[i] << std::endl;
//// ASSERT_NEAR(intersections[0], 0.0, 1e-4, "intersections");
//// ASSERT_NEAR(intersections[1], 0.5, 1e-4, "intersections");
//// ASSERT_NEAR(intersections[2], 0.5, 1e-4, "intersections");
//// ASSERT_NEAR(intersections[3], 1.0, 1e-4, "intersections");
////
//// n = 8;
//// model.intersect_coarse_cell(1, ray, intersections, &n);
//// ASSERT(n == 4, "intersections");
//// ASSERT_NEAR(intersections[0], 0.0, 1e-4, "intersections");
//// ASSERT_NEAR(intersections[1], 0.5, 1e-4, "intersections");
//// ASSERT_NEAR(intersections[2], 0.5, 1e-4, "intersections");
//// ASSERT_NEAR(intersections[3], 1.0, 1e-4, "intersections");
////
//// delete[] intersections;
//// END_TEST_CASE

TEST_SUITE(SpatialPartition)
{
  // TEST_CASE("make_cylindrical_pin_mesh", (test_make_cylindrical_pin_mesh<Float, Int>));
  TEST(makeCoarseCell);
  TEST(makeRTM);
  TEST(makeLattice);
  TEST(makeAssembly);
  TEST(makeAssembly_2d);
  TEST(makeCore);
  TEST(importCoarseCells);
  TEST(io);
  //    TEST_CASE("coarse_cell_face_areas", (test_coarse_cell_face_areas<Float, Int>));
  //    TEST_CASE("coarse_cell_find_face", (test_coarse_cell_find_face<Float, Int>));
  //    TEST_CASE("coarse_cell_ray_intersect", (test_coarse_cell_ray_intersect<Float,
  //    Int>));
}

auto
main() -> int
{
  RUN_SUITE(SpatialPartition);
  return 0;
}
