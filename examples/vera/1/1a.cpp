// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{

  // Parameters
  //  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)

  um2::initialize("debug");
  um2::gmsh::open("1a.brep", /*extra_info=*/true);
  // um2::gmsh::fltk::run();

  //  mpact::SpatialPartition<double, int32_t> model;
  //  model.make_coarse_cell({pitch, pitch});
  //  model.make_rtm({{0}});
  //  model.make_lattice({{0}});
  //  model.make_assembly({0});
  //  model.make_core({{0}});
  //  gmsh::model::occ::overlay_spatial_partition(model);
  //  std::vector<std::pair<int, int>> out;
  //  gmsh::model::getEntities(out, 0);
  //  gmsh::model::mesh::setSize(out, 0.4);
  //  //    gmsh::fltk::run();
  //  gmsh::option::setNumber("Mesh.SecondOrderIncomplete", 1);
  //  gmsh::option::setNumber("Mesh.Algorithm", 5);
  //  gmsh::option::setNumber("Mesh.HighOrderOptimize", 2);
  //  gmsh::model::mesh::generate(2);
  //  gmsh::model::mesh::setOrder(2);
  //  gmsh::model::mesh::optimize("HighOrderElastic");
  //  gmsh::model::mesh::optimize("Relocate2D");
  //  gmsh::model::mesh::optimize("HighOrderElastic");
  //  gmsh::write("1a.inp");
  //  model.import_coarse_cells("1a.inp");
  //  export_mesh("1a.xdmf", model);
  //
  um2::finalize();
  return 0;
}
