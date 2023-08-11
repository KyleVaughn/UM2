// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>
#include "../../helpers.hpp"    
    
auto    
main(int argc, char* argv[]) -> int    
{    
  
  bool const repeated_geom = false;

  um2::MeshType mesh_type = um2::MeshType::None;    
  double lc = 0.0;    
  getGlobalMeshParams(argc, argv, mesh_type, lc);

  double const pitch = 1.26;   // Pitch = 1.26 cm (pg. 4)
  um2::Vec2d const dxdy(pitch, pitch);

  um2::initialize("debug");
  um2::gmsh::open("2a.brep", /*extra_info=*/true);

  um2::mpact::SpatialPartition<double, int32_t> model;

  // Fuel rod and guide tube layout (pg. 5)
  std::vector<std::vector<int>> pin_ids(17, std::vector<int>(17, 0));
  if (!repeated_geom) { 
    for (size_t i = 0; i < 17; ++i) {
      for (size_t j = 0; j < 17; ++j) {
        auto const val = static_cast<int>(i * 17 + j);
        pin_ids[i][j] = val; 
        model.makeCoarseCell(dxdy);
      }
    }
  } else {
    model.makeCoarseCell(dxdy);
  }

  // Construct coarse cells
  model.makeRTM(pin_ids);
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});
  um2::gmsh::model::occ::overlaySpatialPartition(model, "Water");
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);    
  um2::gmsh::model::mesh::generateMesh(mesh_type);
  um2::gmsh::fltk::run();
  um2::gmsh::write("vera_lattice.inp");
  //  model.import_coarse_cells("2a.inp");
  //  export_mesh("2a.xdmf", model);

  um2::finalize();
  return 0;
}
