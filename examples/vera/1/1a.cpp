// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{

  // Parameters
  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)

  um2::initialize();
  um2::gmsh::open("1a.brep", /*extra_info=*/true);
  um2::mpact::SpatialPartition<double, int32_t> model;
  model.makeCoarseCell({pitch, pitch});
  model.makeRTM({{0}});
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});
  um2::gmsh::model::occ::overlaySpatialPartition(model);
  um2::gmsh::model::mesh::setGlobalMeshSize(0.1);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::Tri);
  // um2::gmsh::fltk::run();
  um2::gmsh::write("1a.inp");
  model.importCoarseCells("1a.inp");
  um2::exportMesh("1a.xdmf", model);
  um2::finalize();
  return 0;
}
