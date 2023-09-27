// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>

auto
//NOLINTNEXTLINE
main() -> int
{  
  um2::initialize();
  
  um2::MeshType const mesh_type = um2::MeshType::Tri;
  double const lc = 0.1;

  //create course cells of different sizes to include case of gaps at the edges
  const double pitch = 1.26;   // Pitch = 1.26 cm (pg. 4)
  const double half_gap = 0.4; // Inter-Assembly Half Gap  = 0.04 cm (pg. 7)
  um2::Vec2<double> const dxdy(pitch, pitch);
  um2::Vec2<double> const tall_dxdy(pitch, pitch + half_gap);
  um2::Vec2<double> const wide_dxdy(pitch + half_gap, pitch);
  um2::Vec2<double> const corner_dxdy(pitch + half_gap, pitch + half_gap);

  um2::gmsh::open("2a.brep", /*extra_info=*/true);

  um2::mpact::SpatialPartition model;

  // Fuel rod and guide tube layout (pg. 5)
  // Due to the half gap, the cells on the perimeter are extended
  std::vector<std::vector<Size>> const pin_ids = um2::to_vecvec<Size>(R"(
     7 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 8
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 4
     6 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 1 0 0 1 0 0 2 0 0 1 0 0 1 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 4
     6 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
    10 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 9)");

  // Construct coarse cells
  model.makeCoarseCell(dxdy);        // 0
  model.makeCoarseCell(dxdy);        // 1
  model.makeCoarseCell(dxdy);        // 2
  model.makeCoarseCell(tall_dxdy);   // 3
  model.makeCoarseCell(wide_dxdy);   // 4
  model.makeCoarseCell(tall_dxdy);   // 5
  model.makeCoarseCell(wide_dxdy);   // 6
  model.makeCoarseCell(corner_dxdy); // 7
  model.makeCoarseCell(corner_dxdy); // 8
  model.makeCoarseCell(corner_dxdy); // 9
  model.makeCoarseCell(corner_dxdy); // 10

  //Note: All RTM must be identical in size. Since different coarse cells have 
  //different sizes in this problem, we create a single RTM for all of our course cells
  model.makeRTM(pin_ids);
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});
  um2::gmsh::model::occ::overlaySpatialPartition(model, "Water");
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);
  um2::gmsh::model::mesh::generateMesh(mesh_type);

  // uncomment this line to view the model
  um2::gmsh::fltk::run();
  
  um2::gmsh::write("2a.inp");
  model.importCoarseCells("2a.inp");
  um2::exportMesh("2a.xdmf", model);
  um2::finalize();
  return 0;
}
