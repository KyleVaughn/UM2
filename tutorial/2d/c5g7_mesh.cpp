// BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
// TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
// NEA/NSC/DOC(2001)4

#include <um2.hpp>

int
// NOLINTNEXTLINE(bugprone-exception-escape
main()
{
  um2::initialize();
  um2::gmsh::open("c5g7.brep", /*extra_info=*/true);

  double const lc = 0.1;

  // Construct the MPACT spatial partition using pin-modular ray tracing
  double const pin_pitch = 1.26;               // Pitch = 1.26 cm (pg. 3)
  um2::Vec2d const dxdy(pin_pitch, pin_pitch); // Pin pitch in x and y directions
  um2::mpact::SpatialPartition model;
  // Construct coarse cells
  // 0 UO2               "forestgreen"
  // 1 MOX_4.3           "orange"
  // 2 MOX_7.0           "red3"
  // 3 MOX_8.7           "red4"
  // 4 Fission Chamber   "black"
  // 5 Guide Tube        "darkgrey"
  // 6 Moderator         "royalblue"
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  model.makeCoarseCell(dxdy);
  // Construct RTMs (pin-modular ray tracing)
  model.makeRTM({{0}});
  model.makeRTM({{1}});
  model.makeRTM({{2}});
  model.makeRTM({{3}});
  model.makeRTM({{4}});
  model.makeRTM({{5}});
  model.makeRTM({{6}});
  // Construct lattices
  // UO2 lattice pins (pg. 7)
  std::vector<std::vector<int>> const uo2_rtm_ids = um2::to_vecvec<int>(R"(
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
  std::vector<std::vector<int>> const mox_rtm_ids = um2::to_vecvec<int>(R"(
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
  std::vector<std::vector<int>> const mod_rtm_ids = um2::to_vecvec<int>(R"(
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
  model.makeLattice(uo2_rtm_ids);
  model.makeLattice(mox_rtm_ids);
  model.makeLattice(mod_rtm_ids);
  //   // Make assemblies
  //   size_t const num_axial_segments = 10;
  //   size_t const num_water_segments = num_axial_segments / 10;
  //   size_t const num_fuel_segments = num_axial_segments - num_water_segments;
  //   double const slice_height = model_height / num_axial_segments;
  //   std::vector<double> axial_slice_heights(num_axial_segments + 1);
  //   for (size_t i = 0; i < num_axial_segments + 1; ++i) {
  //     axial_slice_heights[i] = i * slice_height;
  //   }
  //   std::vector<int> uo2_lattice_ids(num_fuel_segments, 0);
  //   uo2_lattice_ids.insert(uo2_lattice_ids.end(), num_water_segments, 2);
  //   std::vector<int> mox_lattice_ids(num_fuel_segments, 1);
  //   mox_lattice_ids.insert(mox_lattice_ids.end(), num_water_segments, 2);
  //   std::vector<int> mod_lattice_ids(num_axial_segments, 2);
  //   model.make_assembly(uo2_lattice_ids, axial_slice_heights);
  //   model.make_assembly(mox_lattice_ids, axial_slice_heights);
  model.makeAssembly({0});
  model.makeAssembly({1});
  model.makeAssembly({2});
  // Make core
  // Core assembly IDs (pg. 6)
  std::vector<std::vector<int>> const core_assembly_ids = um2::to_vecvec<int>(R"(
      0 1 2
      1 0 2
      2 2 2
    )");
  model.makeCore(core_assembly_ids);

  // Overlay the spatial partition
  um2::gmsh::model::occ::overlaySpatialPartition(model);
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);
  um2::gmsh::model::mesh::generate(2);

  um2::gmsh::fltk::run();

  um2::gmsh::write("c5g7.inp");
  model.importCoarseCells("c5g7.inp");
  um2::exportMesh("c5g7.xdmf", model);

  um2::finalize();
  return 0;
}