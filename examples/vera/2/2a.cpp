// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>
#include "../../helpers.hpp"    
    
auto    
main(int argc, char* argv[]) -> int    
{    
  um2::MeshType mesh_type = um2::MeshType::None;    
  double lc = 0.0;    
  getGlobalMeshParams(argc, argv, mesh_type, lc);

  double const pitch = 1.26;   // Pitch = 1.26 cm (pg. 4)
  double const half_gap = 0.4; // Inter-Assembly Half Gap  = 0.04 cm (pg. 7)
  um2::Vec2d const dxdy(pitch, pitch);
  um2::Vec2d const tall_dxdy(pitch, pitch + half_gap);
  um2::Vec2d const wide_dxdy(pitch + half_gap, pitch);
  um2::Vec2d const corner_dxdy(pitch + half_gap, pitch + half_gap);

  um2::initialize("debug");
  um2::gmsh::open("2a.brep", /*extra_info=*/true);

  um2::mpact::SpatialPartition<double, int32_t> model;

  // Fuel rod and guide tube layout (pg. 5)
  // Due to the half gap, the cells on the perimeter are extended
  std::vector<std::vector<int>> const pin_ids = um2::to_vecvec<int>(R"(
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
  model.makeRTM(pin_ids);
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});
  //um2::gmsh::model::occ::overlaySpatialPartition(model, "Water");
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);    
  um2::gmsh::model::mesh::generateMesh(mesh_type);
  um2::gmsh::fltk::run();
  um2::gmsh::write("vera_lattice.inp");
  //  model.import_coarse_cells("2a.inp");
  //  export_mesh("2a.xdmf", model);

  um2::finalize();
  return 0;
}
