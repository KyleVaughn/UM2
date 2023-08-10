// BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
// TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
// NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize("debug");
  um2::gmsh::open("c5g7_pin.brep", /*extra_info=*/true);

  // Construct the MPACT spatial partition using pin-modular ray tracing.
  double const pin_pitch = 1.26; // Pitch = 1.26 cm (pg. 3)
  //  double const assembly_height = 192.78;    // Assembly height = 192.78 cm (pg. 3)
  double const model_height = 214.20;          // Model height = 214.20 cm (pg. 3)
  um2::Vec2d const dxdy(pin_pitch, pin_pitch); // Pin pitch in x and y directions
  um2::mpact::SpatialPartition<double, int32_t> model;
  // 0 UO2
  // 1 Moderator
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  model.makeRTM({{0}});
  model.makeRTM({{1}});
  model.makeLattice({{0}});
  model.makeLattice({{1}});
  // Make assemblies
  // We use the fact that assembly_height = model_height * 0.9
  size_t const num_axial_segments = 10; // Must be a multiple of 10
  size_t const num_water_segments = num_axial_segments / 10;
  size_t const num_fuel_segments = num_axial_segments - num_water_segments;
  double const slice_height = model_height / num_axial_segments;
  std::vector<double> axial_slice_heights(num_axial_segments + 1);
  for (size_t i = 0; i < num_axial_segments + 1; ++i) {
    axial_slice_heights[i] = static_cast<double>(i) * slice_height;
  }
  std::vector<int> uo2_lattice_ids(num_fuel_segments, 0);
  uo2_lattice_ids.insert(uo2_lattice_ids.end(), num_water_segments, 1);
  model.makeAssembly(uo2_lattice_ids, axial_slice_heights);
  model.makeCore({{0}});

  um2::gmsh::model::occ::overlaySpatialPartition(model);
  um2::gmsh::model::mesh::setGlobalMeshSize(0.05);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::fltk::run();
  // um2::gmsh::write("c5g7_pin.inp");
  //   model.import_coarse_cells("c5g7_pin.inp");
  //   export_mesh("c5g7_pin.xdmf", model);
  um2::finalize();
  return 0;
}
