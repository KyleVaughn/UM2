#include <um2/mpact/model.hpp>
#include <um2/stdlib/algorithm/fill.hpp>
#include <um2/common/string_to_lattice.hpp>
#include <um2/common/logger.hpp>

#include "../test_macros.hpp"

#include <numeric> // std::reduce

#include <iostream>

auto constexpr eps = um2::eps_distance;

TEST_CASE(ASCII)
{
  // Increment a string ending in numbers
  um2::String str("00000");
  ASSERT(str == "00000");
  um2::mpact::incrementASCIINumber(str);
  ASSERT(str == "00001");
  str = "_199";
  um2::mpact::incrementASCIINumber(str);
  ASSERT(str == "_200");

  // Create a 5-digit string from an integer
  str = um2::mpact::getASCIINumber(0);
  ASSERT(str == "00000");
  str = um2::mpact::getASCIINumber(199);
  ASSERT(str == "00199");
  str = um2::mpact::getASCIINumber(34578);
  ASSERT(str == "34578");
}

TEST_CASE(addCylindricalPinMesh)
{
  um2::mpact::Model model;
  auto const r0 = castIfNot<Float>(0.4096);
  auto const r1 = castIfNot<Float>(0.475);
  auto const r2 = castIfNot<Float>(0.575);
  um2::Vector<Float> const radii = {r0, r1, r2};
  auto const pitch = castIfNot<Float>(1.26);
  um2::Vector<Int> const num_rings = {3, 1, 1};
  Int const na = 8;

  Int const id = model.addCylindricalPinMesh(pitch, radii, num_rings, na, 1);
  ASSERT(id == 0);
  Int const total_rings = std::reduce(num_rings.begin(), num_rings.end());
  Int const nfaces = (total_rings + 1) * na;
  ASSERT(model.getQuadMesh(0).numFaces() == nfaces);
  auto const aabb = model.getQuadMesh(0).boundingBox();
  ASSERT_NEAR(aabb.extents(0), pitch, eps);
  ASSERT_NEAR(aabb.extents(1), pitch, eps);
  ASSERT_NEAR(aabb.minima(0), 0, eps);
  ASSERT_NEAR(aabb.minima(1), 0, eps);
  Float area_sum = 0;
  for (Int i = 0; i < nfaces; ++i) {
    area_sum += model.getQuadMesh(0).getFace(i).area();
  }
  ASSERT_NEAR(area_sum, pitch * pitch, eps);

  Int const id1 = model.addCylindricalPinMesh(pitch, radii, num_rings, na, 2);
  ASSERT(id1 == 0);
  ASSERT(model.getQuad8Mesh(0).numFaces() == nfaces);
  auto const aabb1 = model.getQuad8Mesh(0).boundingBox();
  ASSERT_NEAR(aabb1.extents(0), pitch, eps);
  ASSERT_NEAR(aabb1.extents(1), pitch, eps);
  ASSERT_NEAR(aabb1.minima(0), 0, eps);
  ASSERT_NEAR(aabb1.minima(1), 0, eps);
  area_sum = 0;
  for (Int i = 0; i < nfaces; ++i) {
    area_sum += model.getQuad8Mesh(0).getFace(i).area();
  }
  ASSERT_NEAR(area_sum, pitch * pitch, eps);
}

TEST_CASE(addRectangularPinMesh)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy(2, 1);

  Int const nx = 2;
  Int const ny = 1;
  Int const id = model.addRectangularPinMesh(dxdy, nx, ny);
  ASSERT(id == 0);
  auto const & mesh = model.getQuadMesh(0);
  ASSERT(mesh.numFaces() == nx * ny);
  auto const aabb = mesh.boundingBox();
  ASSERT(aabb.minima().isApprox(um2::Point2(0, 0)));
  ASSERT(aabb.maxima().isApprox(um2::Point2(2, 1)));

  auto quad = mesh.getFace(0);
  ASSERT_NEAR(quad[0][0], 0, eps);
  ASSERT_NEAR(quad[0][1], 0, eps);
  ASSERT_NEAR(quad[2][0], 1, eps);
  ASSERT_NEAR(quad[2][1], 1, eps);
  quad = mesh.getFace(1);
  ASSERT_NEAR(quad[0][0], 1, eps);
  ASSERT_NEAR(quad[0][1], 0, eps);
  ASSERT_NEAR(quad[2][0], 2, eps);
  ASSERT_NEAR(quad[2][1], 1, eps);
}

TEST_CASE(addCoarseCell)
{
  um2::mpact::Model model;

  // Add a coarse cell that doesn't have a mesh yet
  um2::Vec2F const dxdy(2, 1);
  Int const id = model.addCoarseCell(dxdy);
  ASSERT(id == 0);
  ASSERT(model.numCoarseCells() == 1);

  // Check the properties of the coarse cell
  auto const & cell = model.getCoarseCell(id);
  ASSERT(cell.xy_extents.isApprox(dxdy));
  ASSERT(cell.mesh_type == um2::MeshType::Invalid);
  ASSERT(cell.mesh_id == -1);
  ASSERT(cell.material_ids.empty());

  // Add a cell with full properties
  um2::XSLibrary const lib8(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_8G); 
  model.addRectangularPinMesh(dxdy, 2, 2);
      
  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_51G);    
    
  // Fuel    
  um2::Material fuel;    
  fuel.setName("Fuel");    
  fuel.setDensity(10.42);
  fuel.setTemperature(565.0);
  fuel.setColor(um2::forestgreen);    
  fuel.addNuclide("U234", 6.11864e-6); // Number density in atoms/b-cm    
  fuel.addNuclide("U235", 7.18132e-4);    
  fuel.addNuclide("U236", 3.29861e-6);    
  fuel.addNuclide("U238", 2.21546e-2);    
  fuel.addNuclide("O16", 4.57642e-2);    
  fuel.populateXSec(xslib);

    // Moderator
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743);
  moderator.setTemperature(565.0);
  moderator.setColor(um2::blue);
  moderator.addNuclide("O16", 2.48112e-02);
  moderator.addNuclide("H1", 4.96224e-02);
  moderator.addNuclide("B10", 1.07070e-05);
  moderator.addNuclide("B11", 4.30971e-05);
  moderator.populateXSec(xslib);

  model.addMaterial(fuel);
  model.addMaterial(moderator);
  um2::Vector<MatID> const material_ids = {0, 0, 0, 1};
  Int const id2 =
      model.addCoarseCell(dxdy, um2::MeshType::Quad, 0, material_ids);
  ASSERT(id2 == 1);
  ASSERT(model.numCoarseCells() == 2);

  // Check the properties of the coarse cell
  auto const & cell2 = model.getCoarseCell(id2);
  ASSERT(cell2.xy_extents.isApprox(dxdy));
  ASSERT(cell2.mesh_type == um2::MeshType::Quad);
  ASSERT(cell2.mesh_id == 0);
  ASSERT(cell2.material_ids == material_ids);
}

TEST_CASE(addRTM)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy(2, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);
  ASSERT(model.addCoarseCell(dxdy) == 1);
  ASSERT(model.numCoarseCells() == 2);
  um2::Vector<um2::Vector<Int>> const cc_ids = {
      {0, 1}
  };
  Int id = model.addRTM(cc_ids);
  ASSERT(id == 0);
  ASSERT(model.numRTMs() == 1);
  auto const & rtm = model.getRTM(id);
  ASSERT(rtm.children().size() == 2);
  ASSERT(rtm.children()[0] == 0);
  ASSERT(rtm.children()[1] == 1);
  ASSERT(rtm.grid().numCells(0) == 2);
  ASSERT(rtm.grid().numCells(1) == 1);
  ASSERT_NEAR(rtm.grid().divs(0)[0], 0, eps);
  ASSERT_NEAR(rtm.grid().divs(0)[1], 2, eps);
  ASSERT_NEAR(rtm.grid().divs(0)[2], 4, eps);
  ASSERT_NEAR(rtm.grid().divs(1)[0], 0, eps);
  ASSERT_NEAR(rtm.grid().divs(1)[1], 1, eps);
  model.clear();

  um2::Vector<um2::Vector<Int>> const cc_ids2 = {
      {2, 3},
      {0, 1}
  };
  ASSERT(model.addCoarseCell(dxdy) == 0);
  ASSERT(model.addCoarseCell(dxdy) == 1);
  ASSERT(model.addCoarseCell(dxdy) == 2);
  ASSERT(model.addCoarseCell(dxdy) == 3);
  id = model.addRTM(cc_ids2);
  ASSERT(id == 0);
  ASSERT(model.numRTMs() == 1);
  auto rtm2 = model.getRTM(id);
  ASSERT(rtm2.grid().numCells(0) == 2);
  ASSERT(rtm2.grid().numCells(1) == 2);
  ASSERT(rtm2.getChild(0, 0) == 0);
  ASSERT(rtm2.getChild(1, 0) == 1);
  ASSERT(rtm2.getChild(0, 1) == 2);
  ASSERT(rtm2.getChild(1, 1) == 3);
  ASSERT(rtm2.grid().divs(0).size() == 3);
  ASSERT(rtm2.grid().divs(1).size() == 3);
  ASSERT_NEAR(rtm2.grid().divs(0)[0], 0, eps);
  ASSERT_NEAR(rtm2.grid().divs(0)[1], 2, eps);
  ASSERT_NEAR(rtm2.grid().divs(0)[2], 4, eps);
  ASSERT_NEAR(rtm2.grid().divs(1)[0], 0, eps);
  ASSERT_NEAR(rtm2.grid().divs(1)[1], 1, eps);
  ASSERT_NEAR(rtm2.grid().divs(1)[2], 2, eps);
}

TEST_CASE(addLattice)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy0(3, 3);
  um2::Vec2F const dxdy1(4, 4);
  ASSERT(model.addCoarseCell(dxdy0) == 0);
  ASSERT(model.addCoarseCell(dxdy1) == 1);
  um2::Vector<um2::Vector<Int>> const cc_ids_44 = {
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      {0, 0, 0, 0}
  };
  um2::Vector<um2::Vector<Int>> const cc_ids_33 = {
      {1, 1, 1},
      {1, 1, 1},
      {1, 1, 1}
  };
  ASSERT(model.addRTM(cc_ids_33) == 0);
  ASSERT(model.addRTM(cc_ids_44) == 1);
  um2::Vector<um2::Vector<Int>> const rtm_ids = {
      {0, 1}
  };
  Int const id = model.addLattice(rtm_ids);
  ASSERT(id == 0);
  ASSERT(model.numLattices() == 1);
  auto const lattice = model.getLattice(id);
  ASSERT(lattice.grid().numCells(0) == 2);
  ASSERT(lattice.grid().numCells(1) == 1);
  ASSERT(lattice.getChild(0, 0) == 0);
  ASSERT(lattice.getChild(1, 0) == 1);
  ASSERT_NEAR(lattice.grid().extents(0), 24, eps);
  ASSERT_NEAR(lattice.grid().extents(1), 12, eps);
  ASSERT_NEAR(lattice.grid().minima(0), 0, eps);
  ASSERT_NEAR(lattice.grid().minima(1), 0, eps);
}

TEST_CASE(addAssembly)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy(1, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);
  um2::Vector<um2::Vector<Int>> const cc_ids = {
      {0, 0},
      {0, 0}
  };
  ASSERT(model.addRTM(cc_ids) == 0);
  um2::Vector<um2::Vector<Int>> const rtm_ids = {{0}};
  ASSERT(model.addLattice(rtm_ids) == 0);
  ASSERT(model.addLattice(rtm_ids) == 1);
  um2::Vector<Int> const lat_ids = {0, 1, 0};
  um2::Vector<Float> const lat_z = {0, 2, 3, 4};
  Int const id = model.addAssembly(lat_ids, lat_z);
  ASSERT(id == 0);
  auto const & assembly = model.getAssembly(id);
  ASSERT(model.numAssemblies() == 1);
  ASSERT(assembly.grid().numCells(0) == 3);
  ASSERT(assembly.getChild(0) == 0);
  ASSERT(assembly.getChild(1) == 1);
  ASSERT(assembly.getChild(2) == 0);
  ASSERT(assembly.grid().divs(0).size() == 4);
  ASSERT_NEAR(assembly.grid().divs(0)[0], 0, eps);
  ASSERT_NEAR(assembly.grid().divs(0)[1], 2, eps);
  ASSERT_NEAR(assembly.grid().divs(0)[2], 3, eps);
  ASSERT_NEAR(assembly.grid().divs(0)[3], 4, eps);
}

TEST_CASE(addAssembly_2d)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy(1, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);
  um2::Vector<um2::Vector<Int>> const cc_ids = {
      {0, 0},
      {0, 0}
  };
  ASSERT(model.addRTM(cc_ids) == 0);
  um2::Vector<um2::Vector<Int>> const rtm_ids = {{0}};
  ASSERT(model.addLattice(rtm_ids) == 0);
  ASSERT(model.addLattice(rtm_ids) == 1);
  um2::Vector<Int> const lat_ids = {0};
  Int const id = model.addAssembly(lat_ids);
  ASSERT(id == 0);
  auto const & assembly = model.getAssembly(id);
  ASSERT(model.numAssemblies() == 1);
  ASSERT(assembly.children().size() == 1);
  ASSERT(assembly.children()[0] == 0);
  um2::RectilinearGrid1 const & grid = assembly.grid();
  ASSERT(grid.divs(0).size() == 2);
  auto constexpr ahalf = castIfNot<Float>(0.5);
  ASSERT_NEAR(grid.divs(0)[0], -ahalf, eps);
  ASSERT_NEAR(grid.divs(0)[1], ahalf, eps);
}

TEST_CASE(addCore)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy(2, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);

  um2::Vector<um2::Vector<Int>> const cc_ids = {{0}};
  ASSERT(model.addRTM(cc_ids) == 0);

  um2::Vector<um2::Vector<Int>> const rtm_ids = {{0}};
  ASSERT(model.addLattice(rtm_ids) == 0);

  um2::Vector<Int> const lat_ids1 = {0, 0, 0};
  um2::Vector<Float> const lat_z1 = {0, 2, 3, 4};
  ASSERT(model.addAssembly(lat_ids1, lat_z1) == 0);
  um2::Vector<Int> const lat_ids2 = {0, 0};
  um2::Vector<Float> const lat_z2 = {0, 3, 4};
  ASSERT(model.addAssembly(lat_ids2, lat_z2) == 1);
  ASSERT(model.addAssembly(lat_ids1, lat_z1) == 2);
  ASSERT(model.addAssembly(lat_ids2, lat_z2) == 3);

  um2::Vector<um2::Vector<Int>> const asy_ids = {
      {2, 3},
      {0, 1}
  };
  Int const id = model.addCore(asy_ids);
  ASSERT(id == 0);
  auto const & core = model.getCore();
  ASSERT(core.children().size() == 4);
  ASSERT(core.children()[0] == 0);
  ASSERT(core.children()[1] == 1);
  ASSERT(core.children()[2] == 2);
  ASSERT(core.children()[3] == 3);
  ASSERT(core.grid().divs(0).size() == 3);
  ASSERT(core.grid().divs(1).size() == 3);
  ASSERT_NEAR(core.grid().divs(0)[0], 0, eps);
  ASSERT_NEAR(core.grid().divs(0)[1], 2, eps);
  ASSERT_NEAR(core.grid().divs(0)[2], 4, eps);
  ASSERT_NEAR(core.grid().divs(1)[0], 0, eps);
  ASSERT_NEAR(core.grid().divs(1)[1], 1, eps);
  ASSERT_NEAR(core.grid().divs(1)[2], 2, eps);
}

TEST_CASE(addCoarseGrid)
{
  um2::mpact::Model model;
  um2::Vec2F const dxdy(4, 3);
  um2::Vec2I const nxny(2, 3);
  model.addCoarseGrid(dxdy, nxny);
  ASSERT(model.numCoarseCells() == 6);
  ASSERT(model.numRTMs() == 6);
  ASSERT(model.numLattices() == 1);
  ASSERT(model.numAssemblies() == 1);

  auto const & cc0 = model.getCoarseCell(0);
  ASSERT(cc0.xy_extents.isApprox(um2::Vec2F(2, 1)));
  ASSERT(cc0.mesh_type == um2::MeshType::Invalid);
  ASSERT(cc0.mesh_id == -1);
  ASSERT(cc0.material_ids.empty());
  auto const & cc1 = model.getCoarseCell(1);
  ASSERT(cc1.xy_extents.isApprox(um2::Vec2F(2, 1)));
  ASSERT(cc1.mesh_type == um2::MeshType::Invalid);
  ASSERT(cc1.mesh_id == -1);
  ASSERT(cc1.material_ids.empty());

  auto const & lat = model.getLattice(0);
  ASSERT(lat.grid().numCells(0) == 2);
  ASSERT(lat.grid().numCells(1) == 3);
  ASSERT(lat.getChild(0, 0) == 0);
  ASSERT(lat.getChild(1, 0) == 1);
  ASSERT(lat.getChild(0, 1) == 2);
  ASSERT(lat.getChild(1, 1) == 3);
  ASSERT(lat.getChild(0, 2) == 4);
  ASSERT(lat.getChild(1, 2) == 5);
}

TEST_CASE(importCoarseCellMeshes)
{
  using CoarseCell = typename um2::mpact::Model::CoarseCell;
  um2::mpact::Model model;

  um2::Material clad;
  clad.setName("Clad");
  clad.xsec() = um2::XSec(1);
  clad.xsec().isMacro() = true;

  um2::Material h2o;
  h2o.setName("H2O");
  h2o.xsec() = um2::XSec(1);
  h2o.xsec().isMacro() = true;

  um2::Material uo2;
  uo2.setName("UO2");
  uo2.xsec() = um2::XSec(1);
  uo2.xsec().isMacro() = true;

  model.addMaterial(clad);
  model.addMaterial(h2o);
  model.addMaterial(uo2);

  model.addCoarseCell({1, 1});
  model.addCoarseCell({1, 1});
  model.addCoarseCell({1, 1});
  model.addRTM({
      {2, 2},
      {0, 1}
  });
  model.addLattice({{0}});
  model.addAssembly({0});
  model.addCore({{0}});
  model.importCoarseCellMeshes("./mpact_mesh_files/coarse_cells.inp");

  ASSERT(model.numAssemblies() == 1);
  ASSERT(model.numLattices() == 1);
  ASSERT(model.numRTMs() == 1);
  ASSERT(model.numCoarseCells() == 3);

  CoarseCell const & cell = model.getCoarseCell(0);
  ASSERT(cell.mesh_type == um2::MeshType::Tri);
  ASSERT(cell.mesh_id == 0);
  ASSERT(cell.material_ids.size() == 2);
  ASSERT(cell.material_ids[0] == 1);
  ASSERT(cell.material_ids[1] == 2);
  auto const & tri_mesh = model.getTriMesh(0);
  ASSERT(tri_mesh.numVertices() == 4);
  ASSERT(tri_mesh.getVertex(0).isApprox(um2::Point2(0, 0)));
  ASSERT(tri_mesh.getVertex(1).isApprox(um2::Point2(1, 0)));
  ASSERT(tri_mesh.getVertex(2).isApprox(um2::Point2(1, 1)));
  ASSERT(tri_mesh.getVertex(3).isApprox(um2::Point2(0, 1)));
  ASSERT(tri_mesh.faceVertexConn()[0][0] == 0);
  ASSERT(tri_mesh.faceVertexConn()[0][1] == 1);
  ASSERT(tri_mesh.faceVertexConn()[0][2] == 2);
  ASSERT(tri_mesh.faceVertexConn()[1][0] == 2);
  ASSERT(tri_mesh.faceVertexConn()[1][1] == 3);
  ASSERT(tri_mesh.faceVertexConn()[1][2] == 0);

  CoarseCell const & cell1 = model.getCoarseCell(1);
  ASSERT(cell1.mesh_type == um2::MeshType::Tri);
  ASSERT(cell1.mesh_id == 1);
  ASSERT(cell1.material_ids.size() == 2);
  ASSERT(cell1.material_ids[0] == 1);
  ASSERT(cell1.material_ids[1] == 0);
  auto const & tri_mesh1 = model.getTriMesh(1);
  ASSERT(tri_mesh1.numVertices() == 4);
  ASSERT(tri_mesh1.getVertex(0).isApprox(um2::Point2(0, 0)));
  ASSERT(tri_mesh1.getVertex(1).isApprox(um2::Point2(0, 1)));
  ASSERT(tri_mesh1.getVertex(2).isApprox(um2::Point2(1, 0)));
  ASSERT(tri_mesh1.getVertex(3).isApprox(um2::Point2(1, 1)));
  ASSERT(tri_mesh1.faceVertexConn()[0][0] == 0);
  ASSERT(tri_mesh1.faceVertexConn()[0][1] == 2);
  ASSERT(tri_mesh1.faceVertexConn()[0][2] == 1);
  ASSERT(tri_mesh1.faceVertexConn()[1][0] == 2);
  ASSERT(tri_mesh1.faceVertexConn()[1][1] == 3);
  ASSERT(tri_mesh1.faceVertexConn()[1][2] == 1);

  CoarseCell const & cell2 = model.getCoarseCell(2);
  ASSERT(cell2.mesh_type == um2::MeshType::Quad);
  ASSERT(cell2.mesh_id == 0);
  ASSERT(cell2.material_ids.size() == 1);
  ASSERT(cell2.material_ids[0] == 0);
  auto const & quad_mesh = model.getQuadMesh(0);
  ASSERT(quad_mesh.numVertices() == 4);
  ASSERT(quad_mesh.getVertex(0).isApprox(um2::Point2(1, 0)));
  ASSERT(quad_mesh.getVertex(1).isApprox(um2::Point2(0, 0)));
  ASSERT(quad_mesh.getVertex(2).isApprox(um2::Point2(1, 1)));
  ASSERT(quad_mesh.getVertex(3).isApprox(um2::Point2(0, 1)));
  ASSERT(quad_mesh.faceVertexConn().size() == 1);
  ASSERT(quad_mesh.faceVertexConn()[0][0] == 1);
  ASSERT(quad_mesh.faceVertexConn()[0][1] == 0);
  ASSERT(quad_mesh.faceVertexConn()[0][2] == 2);
  ASSERT(quad_mesh.faceVertexConn()[0][3] == 3);
}

TEST_CASE(operator_PolytopeSoup)
{
  um2::mpact::Model model_out;

  um2::Material clad;
  clad.setName("Clad");
  clad.xsec() = um2::XSec(1);
  clad.xsec().isMacro() = true;

  um2::Material h2o;
  h2o.setName("H2O");
  h2o.xsec() = um2::XSec(1);
  h2o.xsec().isMacro() = true;

  um2::Material uo2;
  uo2.setName("UO2");
  uo2.xsec() = um2::XSec(1);
  uo2.xsec().isMacro() = true;

  model_out.addMaterial(clad);
  model_out.addMaterial(h2o);
  model_out.addMaterial(uo2);

  model_out.addCoarseCell({1, 1});
  model_out.addCoarseCell({1, 1});
  model_out.addCoarseCell({1, 1});
  model_out.addRTM({
      {2, 2},
      {0, 1}
  });
  model_out.addLattice({{0}});
  model_out.addAssembly({0});
  model_out.addCore({{0}});
  model_out.importCoarseCellMeshes("./mpact_mesh_files/coarse_cells.inp");
  um2::PolytopeSoup const soup(model_out);

  // conversion doesn't make mesh manifold, so we expect repeated vertices
  ASSERT(soup.numVertices() == 16);
  ASSERT(soup.getVertex(0).isApprox(um2::Point3(0, 0, 0)));
  ASSERT(soup.getVertex(1).isApprox(um2::Point3(1, 0, 0)));
  ASSERT(soup.getVertex(2).isApprox(um2::Point3(1, 1, 0)));
  ASSERT(soup.getVertex(3).isApprox(um2::Point3(0, 1, 0)));

  ASSERT(soup.getVertex(4).isApprox(um2::Point3(1, 0, 0)));
  ASSERT(soup.getVertex(5).isApprox(um2::Point3(1, 1, 0)));
  ASSERT(soup.getVertex(6).isApprox(um2::Point3(2, 0, 0)));
  ASSERT(soup.getVertex(7).isApprox(um2::Point3(2, 1, 0)));

  ASSERT(soup.getVertex(8).isApprox(um2::Point3(1, 1, 0)));
  ASSERT(soup.getVertex(9).isApprox(um2::Point3(0, 1, 0)));
  ASSERT(soup.getVertex(10).isApprox(um2::Point3(1, 2, 0)));
  ASSERT(soup.getVertex(11).isApprox(um2::Point3(0, 2, 0)));

  ASSERT(soup.getVertex(12).isApprox(um2::Point3(2, 1, 0)));
  ASSERT(soup.getVertex(13).isApprox(um2::Point3(1, 1, 0)));
  ASSERT(soup.getVertex(14).isApprox(um2::Point3(2, 2, 0)));
  ASSERT(soup.getVertex(15).isApprox(um2::Point3(1, 2, 0)));

  ASSERT(soup.numElements() == 6);
  um2::Vector<Int> conn;
  um2::VTKElemType type = um2::VTKElemType::Invalid;
  soup.getElement(0, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn.size() == 3);
  ASSERT(conn[0] == 0);
  ASSERT(conn[1] == 1);
  ASSERT(conn[2] == 2);

  soup.getElement(4, type, conn);
  ASSERT(type == um2::VTKElemType::Quad);
  ASSERT(conn.size() == 4);
  ASSERT(conn[0] == 9);
  ASSERT(conn[1] == 8);
  ASSERT(conn[2] == 10);
  ASSERT(conn[3] == 11);

  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> data;
  soup.getElset(0, name, ids, data);
  ASSERT(name == "Assembly_00000_00000");
  ASSERT(ids.size() == 6);
  ASSERT(ids[0] == 0);
  ASSERT(ids[1] == 1);
  ASSERT(ids[2] == 2);
  ASSERT(ids[3] == 3);
  ASSERT(ids[4] == 4);
  ASSERT(ids[5] == 5);

  soup.getElset(1, name, ids, data);
  ASSERT(name == "Coarse_Cell_00000_00000");
  ASSERT(ids.size() == 2);
  ASSERT(ids[0] == 0);
  ASSERT(ids[1] == 1);

  soup.getElset(2, name, ids, data);
  ASSERT(name == "Coarse_Cell_00001_00000");
  ASSERT(ids.size() == 2);
  ASSERT(ids[0] == 2);
  ASSERT(ids[1] == 3);

  soup.getElset(3, name, ids, data);
  ASSERT(name == "Coarse_Cell_00002_00000");
  ASSERT(ids.size() == 1);
  ASSERT(ids[0] == 4);

  soup.getElset(4, name, ids, data);
  ASSERT(name == "Coarse_Cell_00002_00001");
  ASSERT(ids.size() == 1);
  ASSERT(ids[0] == 5);

}

TEST_CASE(io)
{
  // This is C5G7. We build the model, write it, read it, and check that the 
  // read model is the same as the original model.
  um2::mpact::Model model_out;

  //===========================================================================
  // Materials
  //===========================================================================

  um2::Material uo2;
  uo2.setName("UO2");
  uo2.setColor(um2::forestgreen);
  uo2.xsec() = um2::XSec(1);
  uo2.xsec().isMacro() = true;

  um2::Material mox43;
  mox43.setName("MOX_4.3");
  mox43.setColor(um2::yellow);
  mox43.xsec() = um2::XSec(1);
  mox43.xsec().isMacro() = true;

  um2::Material mox70;
  mox70.setName("MOX_7.0");
  mox70.setColor(um2::orange);
  mox70.xsec() = um2::XSec(1);
  mox70.xsec().isMacro() = true;

  um2::Material mox87;
  mox87.setName("MOX_8.7");
  mox87.setColor(um2::red);
  mox87.xsec() = um2::XSec(1);
  mox87.xsec().isMacro() = true;

  um2::Material fiss_chamber;
  fiss_chamber.setName("Fission_Chamber");
  fiss_chamber.setColor(um2::black);
  fiss_chamber.xsec() = um2::XSec(1);
  fiss_chamber.xsec().isMacro() = true;

  um2::Material guide_tube;
  guide_tube.setName("Guide_Tube");
  guide_tube.setColor(um2::darkgrey);
  guide_tube.xsec() = um2::XSec(1);
  guide_tube.xsec().isMacro() = true;

  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::royalblue);
  moderator.xsec() = um2::XSec(1);
  moderator.xsec().isMacro() = true;

  // Safety checks
  uo2.validateXSec();
  mox43.validateXSec();
  mox70.validateXSec();
  mox87.validateXSec();
  fiss_chamber.validateXSec();
  guide_tube.validateXSec();
  moderator.validateXSec();

  model_out.addMaterial(uo2);
  model_out.addMaterial(mox43);
  model_out.addMaterial(mox70);
  model_out.addMaterial(mox87);
  model_out.addMaterial(fiss_chamber);
  model_out.addMaterial(guide_tube);
  model_out.addMaterial(moderator);

  //===========================================================================
  // Geometry
  //===========================================================================
  
  // Pin meshes
  //---------------------------------------------------------------------------
  auto const radius = castIfNot<Float>(0.54);
  auto const pin_pitch = castIfNot<Float>(1.26);

  um2::Vec2F const xy_extents = {pin_pitch, pin_pitch};

  // Use the same mesh for all pins except the reflector
  um2::Vector<Float> const radii = {radius, castIfNot<Float>(0.62)};
  um2::Vector<Int> const rings = {3, 2};

  // 8 azimuthal divisions, order 2 mesh
  // The first 8 * 3 = 24 faces are the inner material
  // The next 8 * 2 + 8 = 24 faces are moderator
  auto const cyl_pin_mesh_type = um2::MeshType::QuadraticQuad; 
  auto const cyl_pin_id = model_out.addCylindricalPinMesh(pin_pitch, radii, rings, 8, 2);

  // 5 by 5 mesh for the reflector
  auto const rect_pin_mesh_type = um2::MeshType::Quad;
  auto const rect_pin_id = model_out.addRectangularPinMesh(xy_extents, 5, 5);

  // Coarse cells
  //---------------------------------------------------------------------------
  // Pin ID  |  Material
  // --------+----------------
  // 0       |  UO2
  // 1       |  MOX 4.3%
  // 2       |  MOX 7.0%
  // 3       |  MOX 8.7%
  // 4       |  Fission Chamber
  // 5       |  Guide Tube
  // 6       |  Moderator

  // Add the 6 cylindrical pins
  um2::Vector<MatID> mat_ids(48, 6);
  for (MatID i = 0; i < 6; ++i) {
    um2::fill(mat_ids.begin(), mat_ids.begin() + 24, i);
    model_out.addCoarseCell(xy_extents, cyl_pin_mesh_type, cyl_pin_id, mat_ids);
  }

  // Add the 1 rectangular pin
  mat_ids.resize(25);
  um2::fill(mat_ids.begin(), mat_ids.end(), static_cast<MatID>(6)); 
  model_out.addCoarseCell(xy_extents, rect_pin_mesh_type, rect_pin_id, mat_ids); 

  // RTMs
  //---------------------------------------------------------------------------
  // Use pin-modular ray tracing

  um2::Vector<um2::Vector<Int>> ids = {{0}};
  for (Int i = 0; i < 7; ++i) {
    ids[0][0] = i;
    model_out.addRTM(ids);
  }

  // Lattices
  //---------------------------------------------------------------------------

  // UO2 lattice pins (pg. 7)
  um2::Vector<um2::Vector<Int>> const uo2_lattice = um2::stringToLattice<Int>(R"(
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 4 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    )");

  // MOX lattice pins (pg. 7)
  um2::Vector<um2::Vector<Int>> const mox_lattice = um2::stringToLattice<Int>(R"(
      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
      1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
      1 2 2 2 2 5 2 2 5 2 2 5 2 2 2 2 1
      1 2 2 5 2 3 3 3 3 3 3 3 2 5 2 2 1
      1 2 2 2 3 3 3 3 3 3 3 3 3 2 2 2 1
      1 2 5 3 3 5 3 3 5 3 3 5 3 3 5 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 5 3 3 5 3 3 4 3 3 5 3 3 5 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 5 3 3 5 3 3 5 3 3 5 3 3 5 2 1
      1 2 2 2 3 3 3 3 3 3 3 3 3 2 2 2 1
      1 2 2 5 2 3 3 3 3 3 3 3 2 5 2 2 1
      1 2 2 2 2 5 2 2 5 2 2 5 2 2 2 2 1
      1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    )");

    // Moderator lattice
  um2::Vector<um2::Vector<Int>> const h2o_lattice = um2::stringToLattice<Int>(R"( 
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
    )");

  // Ensure the lattices are the correct size
  ASSERT(uo2_lattice.size() == 17);
  ASSERT(uo2_lattice[0].size() == 17);
  ASSERT(mox_lattice.size() == 17);
  ASSERT(mox_lattice[0].size() == 17);
  ASSERT(h2o_lattice.size() == 17);
  ASSERT(h2o_lattice[0].size() == 17);

  model_out.addLattice(uo2_lattice);
  model_out.addLattice(mox_lattice);
  model_out.addLattice(h2o_lattice);

  // Assemblies
  //---------------------------------------------------------------------------
  // Evenly divide into 10 slices
  // The normal model is 60 slices, but use 10 for the test
  // The model is 9 parts fuel, 1 part moderator
  auto const model_height = castIfNot<Float>(214.2);
  auto const num_slices = 10;
  auto const num_fuel_slices = 9 * num_slices / 10;
  um2::Vector<Int> lattice_ids(num_slices, 2); // Fill with H20
  um2::Vector<Float> z_slices(num_slices + 1);
  for (Int i = 0; i <= num_slices; ++i) {
    z_slices[i] = i * model_height / num_slices;
  }

  // uo2 assembly
  um2::fill(lattice_ids.begin(), lattice_ids.begin() + num_fuel_slices, 0);
  model_out.addAssembly(lattice_ids, z_slices);

  // mox assembly
  um2::fill(lattice_ids.begin(), lattice_ids.begin() + num_fuel_slices, 1);
  model_out.addAssembly(lattice_ids, z_slices);

  // moderator assembly 
  um2::fill(lattice_ids.begin(), lattice_ids.begin() + num_slices, 2);
  model_out.addAssembly(lattice_ids, z_slices);

  // Core
  //---------------------------------------------------------------------------
  ids = um2::stringToLattice<Int>(R"(
      0 1 2
      1 0 2
      2 2 2
  )");
  ASSERT(ids.size() == 3);
  ASSERT(ids[0].size() == 3);

  model_out.addCore(ids);

//  model_out.writeOpticalThickness("c5g7_out_optical_thickness.xdmf");

  // Test the numXXXXTotal functions
  ASSERT(model_out.numAssembliesTotal() == 9);
  ASSERT(model_out.numLatticesTotal() == 9 * 10);
  ASSERT(model_out.numRTMsTotal() == 9 * 10 * 17 * 17);
  ASSERT(model_out.numCoarseCellsTotal() == 9 * 10 * 17 * 17);
  // 9 * 17 * 17 * (4 * 48 + 5 * 25) + 1 * 17 * 17 * (9 * 25)
  ASSERT(model_out.numFineCellsTotal() == 889542);

  model_out.write("c5g7_out.xdmf", /*write_knudsen_data=*/false, /*write_xsec_data=*/true);

  um2::mpact::Model model_in;
  model_in.read("c5g7_out.xdmf");

  // Check the materials
  auto const & materials_in = model_in.materials();
  auto const & materials_out = model_out.materials();
  ASSERT(materials_in.size() == 7);
  ASSERT(materials_in[0].getName() == "UO2");
  ASSERT(materials_in[1].getName() == "MOX_4.3");
  ASSERT(materials_in[2].getName() == "MOX_7.0");
  ASSERT(materials_in[3].getName() == "MOX_8.7");
  ASSERT(materials_in[4].getName() == "Fission_Chamber");
  ASSERT(materials_in[5].getName() == "Guide_Tube");
  ASSERT(materials_in[6].getName() == "Moderator");

  for (Int i = 0; i < 7; ++i) {
    ASSERT(materials_in[i].getName() == materials_out[i].getName());
//    for (Int j = 0; j < 7; ++j) {
//      ASSERT_NEAR(materials_in[i].xsec().t(j), materials_out[i].xsec().t(j), eps);
//    }
  }

  // Check the pin meshes. These are duplicated if there are repeated meshes.

  // Check core
  ASSERT(!model_in.core().children().empty());
  ASSERT(model_in.core().grid().numCells(0) == 3);
  ASSERT(model_in.core().grid().numCells(1) == 3);
  ASSERT(model_in.core().getChild(0, 0) == 2);
  ASSERT(model_in.core().getChild(0, 1) == 1);
  ASSERT(model_in.core().getChild(0, 2) == 0);
  ASSERT(model_in.core().getChild(1, 0) == 2);
  ASSERT(model_in.core().getChild(1, 1) == 0);
  ASSERT(model_in.core().getChild(1, 2) == 1);
  ASSERT(model_in.core().getChild(2, 0) == 2);
  ASSERT(model_in.core().getChild(2, 1) == 2);
  ASSERT(model_in.core().getChild(2, 2) == 2);
}

//TEST_CASE(getCoarseCellOpticalThickness)
//{
//
//  um2::mpact::Model model;
//
//  //===========================================================================
//  // Materials
//  //===========================================================================
//
//  um2::Material uo2;
//  uo2.setName("UO2");
//  uo2.setColor(um2::forestgreen);
//  uo2.xsec().t() = {2.12450e-01, 3.55470e-01, 4.85540e-01, 5.59400e-01,
//                    3.18030e-01, 4.01460e-01, 5.70610e-01};
//  uo2.xsec().isMacro() = true;
//
//  um2::Material moderator;
//  moderator.setName("Moderator");
//  moderator.setColor(um2::royalblue);
//  moderator.xsec().t() = {2.30070e-01, 7.76460e-01, 1.48420e+00,
// 1.50520e+00, 1.55920e+00, 2.02540e+00,
// 3.30570e+00};
//  moderator.xsec().isMacro() = true;
//
//  uo2.validateXSec();
//  moderator.validateXSec();
//
//  model.addMaterial(uo2);
//  model.addMaterial(moderator);
//
//  //===========================================================================
//  // Geometry
//  //===========================================================================
//  
//  auto const pin_pitch = castIfNot<Float>(1.26);
//  um2::Vec2F const xy_extents = {pin_pitch, pin_pitch};
//                                                                                          
//  auto const rect_pin_mesh_type = um2::MeshType::Quad;
//  auto const rect_pin_id = model.addRectangularPinMesh(xy_extents, 2, 1);
//                                                                                          
//  // Coarse cells
//  //---------------------------------------------------------------------------
// 
//  // Add a pure fuel pin cell
//  um2::Vector<MatID> mat_ids(2, 0);
//  model.addCoarseCell(xy_extents, rect_pin_mesh_type, rect_pin_id, mat_ids);
//
//  // Compute the optical thickness for each energy group
//  um2::Vector<Float> taus(7);
//  model.getCoarseCellOpticalThickness(0, taus);
//
//  // In a homogenous cell, the optical thickness is simply:
//  // tau_g = sigma_t,g * mcl
//  // where mcl is the mean chord length of the cell
//  // mcl = pi * area / perimeter
//
//  auto const area = pin_pitch * pin_pitch;
//  auto const perimeter = 4 * pin_pitch; 
//  auto const mcl = um2::pi<Float> * area / perimeter;
//  
//  for (Int g = 0; g < 7; ++g) {
//    Float const sigma_t0 = uo2.xsec().t(g);
//    Float const tau_ref = sigma_t0 * mcl;
//    Float const rel_err = (taus[g] - tau_ref) / tau_ref;
//    ASSERT_NEAR(rel_err, 0, eps);
//  }
//
//  // Add a pin with one face fuel, one face moderator
//  mat_ids[1] = 1;
//  model.addCoarseCell(xy_extents, rect_pin_mesh_type, rect_pin_id, mat_ids);
//  um2::fill(taus.begin(), taus.end(), static_cast<Float>(0)); 
//  model.getCoarseCellOpticalThickness(1, taus);
//  
//  for (Int g = 0; g < 7; ++g) {
//    Float const sigma_t0 = uo2.xsec().t(g);
//    Float const sigma_t1 = moderator.xsec().t(g);
//    Float const tau_ref = (sigma_t0 * mcl + sigma_t1 * mcl) / 2; 
//    Float const rel_err = (taus[g] - tau_ref) / tau_ref;
//    ASSERT_NEAR(rel_err, 0, eps);
//  }
//
//  // Add a pin like the following:
//  // 0 0 1 0 0
//  // 0 0 1 0 0
//  // 0 0 1 1 1
//  // 0 0 0 0 1
//  // 0 0 0 0 1
//  auto const rect_pin_id_5x5 = model.addRectangularPinMesh(xy_extents, 5, 5);
//  mat_ids.resize(25);
//  um2::fill(mat_ids.begin(), mat_ids.end(), static_cast<MatID>(0));
//  mat_ids[4] = static_cast<MatID>(1);
//  mat_ids[9] = static_cast<MatID>(1);
//  mat_ids[12] = static_cast<MatID>(1);
//  mat_ids[13] = static_cast<MatID>(1);
//  mat_ids[14] = static_cast<MatID>(1);
//  mat_ids[17] = static_cast<MatID>(1);
//  mat_ids[22] = static_cast<MatID>(1);
//
//  model.addCoarseCell(xy_extents, rect_pin_mesh_type, rect_pin_id_5x5, mat_ids);
//  model.getCoarseCellOpticalThickness(2, taus);
//
//  for (Int g = 0; g < 7; ++g) {
//    Float const sigma_t0 = uo2.xsec().t(g);
//    Float const sigma_t1 = moderator.xsec().t(g);
//    Float const area0 = 18 * area / 25;
//    Float const area1 = 7 * area / 25;
//    Float const tau_ref = um2::pi<Float> * (sigma_t0 * area0 + sigma_t1 * area1) / perimeter; 
//    Float const rel_err = (taus[g] - tau_ref) / tau_ref;
//    ASSERT_NEAR(rel_err, 0, eps);
//  }
//}

TEST_SUITE(mpact_Model)
{
  TEST(ASCII);
  TEST(addCylindricalPinMesh);
  TEST(addRectangularPinMesh);
  TEST(addCoarseCell);
  TEST(addRTM);
  TEST(addLattice);
  TEST(addAssembly);
  TEST(addAssembly_2d);
  TEST(addCore);
  TEST(addCoarseGrid);
  TEST(importCoarseCellMeshes);
  TEST(operator_PolytopeSoup);
 // TEST(io);
//  TEST(getCoarseCellOpticalThickness);
}

auto
main() -> int
{
  RUN_SUITE(mpact_Model);
  return 0;
}
