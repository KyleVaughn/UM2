#include <um2/mpact/spatial_partition.hpp>

#include "../test_macros.hpp"

#include <numeric> // std::reduce

F constexpr eps = condCast<F>(1e-6);

TEST_CASE(addCylindricalPinMesh)
{
  um2::mpact::SpatialPartition model;
  F const r0 = condCast<F>(0.4096);
  F const r1 = condCast<F>(0.475);
  F const r2 = condCast<F>(0.575);
  um2::Vector<F> const radii = {r0, r1, r2};
  F const pitch = condCast<F>(1.26);
  um2::Vector<I> const num_rings = {3, 1, 1};
  I const na = 8;

  I const id = model.addCylindricalPinMesh(radii, pitch, num_rings, na, 1);
  ASSERT(id == 0);
  I const total_rings = std::reduce(num_rings.begin(), num_rings.end());
  I const nfaces = (total_rings + 1) * na;
  ASSERT(model.getQuadMesh(0).numFaces() == nfaces);
  auto const aabb = model.getQuadMesh(0).boundingBox();
  ASSERT_NEAR(aabb.width(), pitch, eps);
  ASSERT_NEAR(aabb.height(), pitch, eps);
  ASSERT_NEAR(aabb.minima(0), 0, eps);
  ASSERT_NEAR(aabb.minima(1), 0, eps);

  I const id1 = model.addCylindricalPinMesh(radii, pitch, num_rings, na, 2);
  ASSERT(id1 == 0);
  ASSERT(model.getQuad8Mesh(0).numFaces() == nfaces);
  auto const aabb1 = model.getQuad8Mesh(0).boundingBox();
  ASSERT_NEAR(aabb1.width(), pitch, eps);
  ASSERT_NEAR(aabb1.height(), pitch, eps);
  ASSERT_NEAR(aabb1.minima(0), 0, eps);
  ASSERT_NEAR(aabb1.minima(1), 0, eps);

  um2::Vector<MaterialID> material_ids;
  um2::Material uo2;
  uo2.setName("UO2"); 
  uo2.setColor(um2::forestgreen);
  um2::Material clad;
  clad.setName("Clad");
  clad.setColor(um2::lightgray);
  um2::Material water;
  water.setName("Water");
  water.setColor(um2::lightblue);
  model.addMaterial(uo2);
  model.addMaterial(clad);
  model.addMaterial(water);
  material_ids.push_back(num_rings[0] * na, 0);
  material_ids.push_back(num_rings[1] * na, 1);
  material_ids.push_back((num_rings[2] + 1) * na, 2);
  I const id2 =
      model.addCoarseCell({pitch, pitch}, um2::MeshType::QuadraticQuad, 0, material_ids);
  ASSERT(id2 == 0);
  auto const & coarse_cell = model.getCoarseCell(0);
  ASSERT_NEAR(coarse_cell.dxdy[0], pitch, eps);
  ASSERT_NEAR(coarse_cell.dxdy[1], pitch, eps);
  ASSERT(coarse_cell.mesh_type == um2::MeshType::QuadraticQuad);
  ASSERT(coarse_cell.mesh_id == 0);
  ASSERT(coarse_cell.material_ids == material_ids);
}

TEST_CASE(addRectangularPinMesh)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> dxdy(2, 1);

  I id = -1;
  I const nx = 2;
  I const ny = 1;
  id = model.addRectangularPinMesh(dxdy, nx, ny);
  ASSERT(id == 0);
  auto const & mesh = model.getQuadMesh(0);
  ASSERT(mesh.numFaces() == nx * ny);
  auto const aabb = mesh.boundingBox();
  ASSERT_NEAR(aabb.width(), dxdy[0], eps);
  ASSERT_NEAR(aabb.height(), dxdy[1], eps);
  ASSERT_NEAR(aabb.minima(0), 0, eps);
  ASSERT_NEAR(aabb.minima(1), 0, eps);

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
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> const dxdy(2, 1);
  I const id = model.addCoarseCell(dxdy);
  ASSERT(id == 0);
  ASSERT(model.numCoarseCells() == 1);
  auto const & cell = model.getCoarseCell(id);
  ASSERT(um2::isApprox(cell.dxdy, dxdy));
  ASSERT(cell.mesh_type == um2::MeshType::None);
  ASSERT(cell.mesh_id == -1);
  ASSERT(cell.material_ids.empty());
  I const id2 = model.addCoarseCell(dxdy);
  ASSERT(id2 == 1);
  auto const & cell2 = model.getCoarseCell(id2);
  ASSERT(model.numCoarseCells() == 2);
  ASSERT(um2::isApprox(cell2.dxdy, dxdy));
  ASSERT(cell2.mesh_type == um2::MeshType::None);
  ASSERT(cell2.mesh_id == -1);
  ASSERT(cell2.material_ids.empty());
}

TEST_CASE(addRTM)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> const dxdy(2, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);
  ASSERT(model.addCoarseCell(dxdy) == 1);
  ASSERT(model.numCoarseCells() == 2);
  um2::Vector<um2::Vector<I>> const cc_ids = {
      {0, 1}
  };
  I id = model.addRTM(cc_ids);
  ASSERT(id == 0);
  ASSERT(model.numRTMs() == 1);
  auto rtm = model.getRTM(id);
  ASSERT(rtm.children().size() == 2);
  ASSERT(rtm.children()[0] == 0);
  ASSERT(rtm.children()[1] == 1);
  ASSERT(rtm.grid().numXCells() == 2);
  ASSERT(rtm.grid().numYCells() == 1);
  ASSERT_NEAR(rtm.grid().divs(0)[0], 0, eps);
  ASSERT_NEAR(rtm.grid().divs(0)[1], 2, eps);
  ASSERT_NEAR(rtm.grid().divs(0)[2], 4, eps);
  ASSERT_NEAR(rtm.grid().divs(1)[0], 0, eps);
  ASSERT_NEAR(rtm.grid().divs(1)[1], 1, eps);
  model.clear();

  um2::Vector<um2::Vector<I>> const cc_ids2 = {
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
  ASSERT(rtm2.grid().numXCells() == 2);
  ASSERT(rtm2.grid().numYCells() == 2);
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
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> const dxdy0(3, 3);
  um2::Vec2<F> const dxdy1(4, 4);
  ASSERT(model.addCoarseCell(dxdy0) == 0);
  ASSERT(model.addCoarseCell(dxdy1) == 1);
  um2::Vector<um2::Vector<I>> const cc_ids_44 = {
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      {0, 0, 0, 0}
  };
  um2::Vector<um2::Vector<I>> const cc_ids_33 = {
      {1, 1, 1},
      {1, 1, 1},
      {1, 1, 1}
  };
  ASSERT(model.addRTM(cc_ids_33) == 0);
  ASSERT(model.addRTM(cc_ids_44) == 1);
  um2::Vector<um2::Vector<I>> const rtm_ids = {
      {0, 1}
  };
  I const id = model.addLattice(rtm_ids);
  ASSERT(id == 0);
  ASSERT(model.numLattices() == 1);
  auto const lattice = model.getLattice(id);
  ASSERT(lattice.grid().numXCells() == 2);
  ASSERT(lattice.grid().numYCells() == 1);
  ASSERT(lattice.getChild(0, 0) == 0);
  ASSERT(lattice.getChild(1, 0) == 1);
  ASSERT_NEAR(lattice.grid().dx(), 12, eps);
  ASSERT_NEAR(lattice.grid().dy(), 12, eps);
  ASSERT_NEAR(lattice.grid().xMin(), 0, eps);
  ASSERT_NEAR(lattice.grid().yMin(), 0, eps);
}

TEST_CASE(addAssembly)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> const dxdy(1, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);
  um2::Vector<um2::Vector<I>> const cc_ids = {
      {0, 0},
      {0, 0}
  };
  ASSERT(model.addRTM(cc_ids) == 0);
  um2::Vector<um2::Vector<I>> const rtm_ids = {{0}};
  ASSERT(model.addLattice(rtm_ids) == 0);
  ASSERT(model.addLattice(rtm_ids) == 1);
  um2::Vector<I> const lat_ids = {0, 1, 0};
  um2::Vector<F> const lat_z = {0, 2, 3, 4};
  I const id = model.addAssembly(lat_ids, lat_z);
  ASSERT(id == 0);
  auto const & assembly = model.getAssembly(id);
  ASSERT(model.numAssemblies() == 1);
  ASSERT(assembly.grid().numXCells() == 3);
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
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> const dxdy(1, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);
  um2::Vector<um2::Vector<I>> const cc_ids = {
      {0, 0},
      {0, 0}
  };
  ASSERT(model.addRTM(cc_ids) == 0);
  um2::Vector<um2::Vector<I>> const rtm_ids = {{0}};
  ASSERT(model.addLattice(rtm_ids) == 0);
  ASSERT(model.addLattice(rtm_ids) == 1);
  um2::Vector<I> const lat_ids = {0};
  I const id = model.addAssembly(lat_ids);
  ASSERT(id == 0);
  auto const & assembly = model.getAssembly(id);
  ASSERT(model.numAssemblies() == 1);
  ASSERT(assembly.children().size() == 1);
  ASSERT(assembly.children()[0] == 0);
  um2::RectilinearGrid1 const & grid = assembly.grid();
  ASSERT(grid.divs(0).size() == 2);
  ASSERT_NEAR(grid.divs(0)[0], -1, eps);
  ASSERT_NEAR(grid.divs(0)[1], 1, eps);
}

TEST_CASE(addCore)
{
  um2::mpact::SpatialPartition model;
  um2::Vec2<F> const dxdy(2, 1);
  ASSERT(model.addCoarseCell(dxdy) == 0);

  um2::Vector<um2::Vector<I>> const cc_ids = {{0}};
  ASSERT(model.addRTM(cc_ids) == 0);

  um2::Vector<um2::Vector<I>> const rtm_ids = {{0}};
  ASSERT(model.addLattice(rtm_ids) == 0);

  um2::Vector<I> const lat_ids1 = {0, 0, 0};
  um2::Vector<F> const lat_z1 = {0, 2, 3, 4};
  ASSERT(model.addAssembly(lat_ids1, lat_z1) == 0);
  um2::Vector<I> const lat_ids2 = {0, 0};
  um2::Vector<F> const lat_z2 = {0, 3, 4};
  ASSERT(model.addAssembly(lat_ids2, lat_z2) == 1);
  ASSERT(model.addAssembly(lat_ids1, lat_z1) == 2);
  ASSERT(model.addAssembly(lat_ids2, lat_z2) == 3);

  um2::Vector<um2::Vector<I>> const asy_ids = {
      {2, 3},
      {0, 1}
  };
  I const id = model.addCore(asy_ids);
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

TEST_CASE(importCoarseCells)
{
  using CoarseCell = typename um2::mpact::SpatialPartition::CoarseCell;
  um2::mpact::SpatialPartition model;
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
  model.importCoarseCells("./mpact_mesh_files/coarse_cells.inp");

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
  ASSERT(um2::isApprox(tri_mesh.getVertex(0), {0, 0}));
  ASSERT(um2::isApprox(tri_mesh.getVertex(1), {1, 0}));
  ASSERT(um2::isApprox(tri_mesh.getVertex(2), {1, 1}));
  ASSERT(um2::isApprox(tri_mesh.getVertex(3), {0, 1}));
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
  ASSERT(um2::isApprox(tri_mesh1.getVertex(0), {0, 0}));
  ASSERT(um2::isApprox(tri_mesh1.getVertex(1), {0, 1}));
  ASSERT(um2::isApprox(tri_mesh1.getVertex(2), {1, 0}));
  ASSERT(um2::isApprox(tri_mesh1.getVertex(3), {1, 1}));
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
  ASSERT(um2::isApprox(quad_mesh.getVertex(0), {1, 0}));
  ASSERT(um2::isApprox(quad_mesh.getVertex(1), {0, 0}));
  ASSERT(um2::isApprox(quad_mesh.getVertex(2), {1, 1}));
  ASSERT(um2::isApprox(quad_mesh.getVertex(3), {0, 1}));
  ASSERT(quad_mesh.faceVertexConn().size() == 1);
  ASSERT(quad_mesh.faceVertexConn()[0][0] == 1);
  ASSERT(quad_mesh.faceVertexConn()[0][1] == 0);
  ASSERT(quad_mesh.faceVertexConn()[0][2] == 2);
  ASSERT(quad_mesh.faceVertexConn()[0][3] == 3);
}
//////
////// template <typename T, typename I>
////// TEST_CASE(toPolytopeSoup)
//////{
//////   //  using CoarseCell = typename um2::mpact::SpatialPartition::CoarseCell;
//////   um2::mpact::SpatialPartition model_out;
//////   model_out.addCoarseCell({1, 1});
//////   model_out.addCoarseCell({1, 1});
//////   model_out.addCoarseCell({1, 1});
//////   model_out.addRTM({
//////       {2, 2},
//////       {0, 1}
//////   });
//////   model_out.addLattice({{0}});
//////   model_out.addAssembly({0});
//////   model_out.addCore({{0}});
//////   model_out.importCoarseCells("./mpact_mesh_files/coarse_cells.inp");
//////   um2::PolytopeSoup soup;
//////   model_out.toPolytopeSoup(soup);
//////   soup.write("./mpact_export_test_model.xdmf");
//////
//////   //  um2::exportMesh(filepath, model_out);
//////   //  um2::mpact::SpatialPartition model;
//////   //  um2::importMesh(filepath, model);
//////   //
//////   //  ASSERT(model.numAssemblies() == 1);
//////   //  ASSERT(model.numLattices() == 1);
//////   //  ASSERT(model.numRTMs() == 1);
//////   //  ASSERT(model.numCoarseCells() == 3);
//////   //
//////   //  ASSERT(model.tri.size() == 2);
//////   //  CoarseCell const & cell = model.coarse_cells[0];
//////   //  ASSERT(cell.mesh_type == um2::MeshType::Tri);
//////   //  ASSERT(cell.mesh_id == 0);
//////   //  ASSERT(cell.material_ids.size() == 2);
//////   //  ASSERT(cell.material_ids[0] == 1);
//////   //  ASSERT(cell.material_ids[1] == 2);
//////   //  um2::TriMesh<2, T, I> const & tri_mesh = model.tri[0];
//////   //  ASSERT(tri_mesh.numVertices() == 4);
//////   //  ASSERT(um2::isApprox(tri_mesh.vertices[0], {0, 0}));
//////   //  ASSERT(um2::isApprox(tri_mesh.vertices[1], {1, 0}));
//////   //  ASSERT(um2::isApprox(tri_mesh.vertices[2], {1, 1}));
//////   //  ASSERT(um2::isApprox(tri_mesh.vertices[3], {0, 1}));
//////   //  ASSERT(tri_mesh.fv[0][0] == 0);
//////   //  ASSERT(tri_mesh.fv[0][1] == 1);
//////   //  ASSERT(tri_mesh.fv[0][2] == 2);
//////   //  ASSERT(tri_mesh.fv[1][0] == 2);
//////   //  ASSERT(tri_mesh.fv[1][1] == 3);
//////   //  ASSERT(tri_mesh.fv[1][2] == 0);
//////   //
//////   //  CoarseCell const & cell1 = model.coarse_cells[1];
//////   //  ASSERT(cell1.mesh_type == um2::MeshType::Tri);
//////   //  ASSERT(cell1.mesh_id == 1);
//////   //  ASSERT(cell1.material_ids.size() == 2);
//////   //  ASSERT(cell1.material_ids[0] == 1);
//////   //  ASSERT(cell1.material_ids[1] == 0);
//////   //  um2::TriMesh<2, T, I> const & tri_mesh1 = model.tri[1];
//////   //  ASSERT(tri_mesh1.vertices.size() == 4);
//////   //  ASSERT(um2::isApprox(tri_mesh1.vertices[0], {0, 0}));
//////   //  ASSERT(um2::isApprox(tri_mesh1.vertices[1], {0, 1}));
//////   //  ASSERT(um2::isApprox(tri_mesh1.vertices[2], {1, 0}));
//////   //  ASSERT(um2::isApprox(tri_mesh1.vertices[3], {1, 1}));
//////   //  ASSERT(tri_mesh1.fv[0][0] == 0);
//////   //  ASSERT(tri_mesh1.fv[0][1] == 2);
//////   //  ASSERT(tri_mesh1.fv[0][2] == 1);
//////   //  ASSERT(tri_mesh1.fv[1][0] == 2);
//////   //  ASSERT(tri_mesh1.fv[1][1] == 3);
//////   //  ASSERT(tri_mesh1.fv[1][2] == 1);
//////   //
//////   //  CoarseCell const & cell2 = model.coarse_cells[2];
//////   //  ASSERT(cell2.mesh_type == um2::MeshType::Quad);
//////   //  ASSERT(cell2.mesh_id == 0);
//////   //  ASSERT(cell2.material_ids.size() == 1);
//////   //  ASSERT(cell2.material_ids[0] == 0);
//////   //  um2::QuadMesh<2, T, I> const & quad_mesh = model.quad[0];
//////   //  ASSERT(quad_mesh.vertices.size() == 4);
//////   //  ASSERT(um2::isApprox(quad_mesh.vertices[0], {1, 0}));
//////   //  ASSERT(um2::isApprox(quad_mesh.vertices[1], {0, 0}));
//////   //  ASSERT(um2::isApprox(quad_mesh.vertices[2], {1, 1}));
//////   //  ASSERT(um2::isApprox(quad_mesh.vertices[3], {0, 1}));
//////   //  ASSERT(quad_mesh.fv.size() == 1);
//////   //  ASSERT(quad_mesh.fv[0][0] == 1);
//////   //  ASSERT(quad_mesh.fv[0][1] == 0);
//////   //  ASSERT(quad_mesh.fv[0][2] == 2);
//////   //  ASSERT(quad_mesh.fv[0][3] == 3);
//////   //
//////   //  int stat = std::remove("./mpact_export_test_model.xdmf");
//////   //  ASSERT(stat == 0);
//////   //  stat = std::remove("./mpact_export_test_model.h5");
//////   //  ASSERT(stat == 0);
////// }
//////////// template <typename T, typename I>
//////////// TEST_CASE(test_coarse_cell_face_areas)
//////////// um2::mpact::SpatialPartition model;
//////////// model.addCoarseCell({1, 1});
//////////// model.addCoarseCell({1, 1});
//////////// model.addCoarseCell({1, 1});
//////////// model.make_rtm({
////////////     {2, 2},
////////////     {0, 1}
//////////// });
//////////// model.make_lattice({{0}});
//////////// model.make_assembly({0});
//////////// model.make_core({{0}});
//////////// model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
////////////
//////////// um2::Vector<F> areas;
//////////// model.coarse_cell_face_areas(0, areas);
//////////// ASSERT(areas.size() == 2, "areas");
//////////// ASSERT_NEAR(areas[0], 0.5, 1e-4, "areas");
//////////// ASSERT_NEAR(areas[1], 0.5, 1e-4, "areas");
//////////// model.coarse_cell_face_areas(1, areas);
//////////// ASSERT(areas.size() == 2, "areas");
//////////// ASSERT_NEAR(areas[0], 0.5, 1e-4, "areas");
//////////// ASSERT_NEAR(areas[1], 0.5, 1e-4, "areas");
//////////// model.coarse_cell_face_areas(2, areas);
//////////// ASSERT(areas.size() == 1, "areas");
//////////// ASSERT_NEAR(areas[0], 1.0, 1e-4, "areas");
//////////// END_TEST_CASE
////////////
//////////// template <typename T, typename I>
//////////// TEST_CASE(test_coarse_cell_find_face)
//////////// um2::mpact::SpatialPartition model;
//////////// model.addCoarseCell({1, 1});
//////////// model.addCoarseCell({1, 1});
//////////// model.addCoarseCell({1, 1});
//////////// model.make_rtm({
////////////     {2, 2},
////////////     {0, 1}
//////////// });
//////////// model.make_lattice({{0}});
//////////// model.make_assembly({0});
//////////// model.make_core({{0}});
//////////// model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
////////////
//////////// length_t face_id = model.coarse_cell_find_face(
////////////     2, um2::Point2<F>(condCast<F>(0.5), condCast<F>(0.5)));
//////////// ASSERT(face_id == 0, "face_id");
//////////// face_id = -2;
//////////// face_id = model.coarse_cell_find_face(
////////////     2, um2::Point2<F>(condCast<F>(0.5), condCast<F>(1.5)));
//////////// ASSERT(face_id == -1, "face_id");
//////////// face_id = -2;
////////////
//////////// face_id = model.coarse_cell_find_face(
////////////     1, um2::Point2<F>(condCast<F>(0.5), condCast<F>(0.05)));
//////////// ASSERT(face_id == 0, "face_id");
//////////// face_id = -2;
//////////// face_id = model.coarse_cell_find_face(
////////////     1, um2::Point2<F>(condCast<F>(0.5), condCast<F>(-0.05)));
//////////// ASSERT(face_id == -1, "face_id");
//////////// face_id = -2;
//////////// face_id = model.coarse_cell_find_face(
////////////     1, um2::Point2<F>(condCast<F>(0.5), condCast<F>(0.95)));
//////////// ASSERT(face_id == 1, "face_id");
//////////// face_id = -2;
//////////// END_TEST_CASE
////////////
//////////// template <typename T, typename I>
//////////// TEST_CASE(test_coarse_cell_ray_intersect)
//////////// um2::mpact::SpatialPartition model;
//////////// model.addCoarseCell({1, 1});
//////////// model.addCoarseCell({1, 1});
//////////// model.addCoarseCell({1, 1});
//////////// model.make_rtm({
////////////     {2, 2},
////////////     {0, 1}
//////////// });
//////////// model.make_lattice({{0}});
//////////// model.make_assembly({0});
//////////// model.make_core({{0}});
//////////// model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
////////////
//////////// um2::Ray2<F> ray(um2::Point2<F>(condCast<F>(0),
/////////// condCast<F>(0.5)), /                  um2::Vec2<F>(1, 0)); / int n = 8; /
///////// T
///////////* intersections = new T[n]; / model.intersect_coarse_cell(0, ray, intersections,
///////&n);
//////////// ASSERT(n == 4, "intersections");
//////////// for (int i = 0; i < n; i++)
////////////   std::cout << intersections[i] << std::endl;
//////////// ASSERT_NEAR(intersections[0], 0.0, 1e-4, "intersections");
//////////// ASSERT_NEAR(intersections[1], 0.5, 1e-4, "intersections");
//////////// ASSERT_NEAR(intersections[2], 0.5, 1e-4, "intersections");
//////////// ASSERT_NEAR(intersections[3], 1.0, 1e-4, "intersections");
////////////
//////////// n = 8;
//////////// model.intersect_coarse_cell(1, ray, intersections, &n);
//////////// ASSERT(n == 4, "intersections");
//////////// ASSERT_NEAR(intersections[0], 0.0, 1e-4, "intersections");
//////////// ASSERT_NEAR(intersections[1], 0.5, 1e-4, "intersections");
//////////// ASSERT_NEAR(intersections[2], 0.5, 1e-4, "intersections");
//////////// ASSERT_NEAR(intersections[3], 1.0, 1e-4, "intersections");
////////////
//////////// delete[] intersections;
//////////// END_TEST_CASE

TEST_SUITE(SpatialPartition)
{
  TEST(addCylindricalPinMesh);
  TEST(addRectangularPinMesh);
  TEST(addCoarseCell);
  TEST(addRTM);
  TEST(addLattice);
  TEST(addAssembly);
  TEST(addAssembly_2d);
  TEST(addCore);
  TEST(importCoarseCells);
  //  TEST((toPolytopeSoup));
  //    TEST_CASE("coarse_cell_face_areas", (test_coarse_cell_face_areas));
  //    TEST_CASE("coarse_cell_find_face", (test_coarse_cell_find_face));
  //    TEST_CASE("coarse_cell_ray_intersect", (test_coarse_cell_ray_intersect<T,
  //    I>));
}

auto
main() -> int
{
  RUN_SUITE(SpatialPartition);
  return 0;
}
