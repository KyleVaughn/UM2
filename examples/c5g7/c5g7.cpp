// BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
// TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
// NEA/NSC/DOC(2001)4

#include <um2.hpp>

template <typename T>
using vecvec = std::vector<std::vector<T>>;

int
main(int argc, char ** argv)
{

  using namespace um2;

  initialize();
  gmsh::open("c5g7.brep", true);

  // Construct the MPACT spatial partition using pin-modular ray tracing.
  double const pin_pitch = 1.26; // Pitch = 1.26 cm (pg. 3)
  //    double const assembly_height = 192.78;    // Assembly height = 192.78 cm (pg. 3)
  double const model_height = 214.20;     // Model height = 214.20 cm (pg. 3)
  Vec2d const dxdy(pin_pitch, pin_pitch); // Pin pitch in x and y directions
  mpact::SpatialPartition<double, int32_t> model;
  // Construct coarse cells
  // 0 UO2               "forestgreen"
  // 1 MOX_4.3           "orange"
  // 2 MOX_7.0           "red3"
  // 3 MOX_8.7           "red4"
  // 4 Fission Chamber   "black"
  // 5 Guide Tube        "darkgrey"
  // 6 Moderator         "royalblue"
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  // Construct RTMs (pin-modular ray tracing)
  model.make_rtm({{0}});
  model.make_rtm({{1}});
  model.make_rtm({{2}});
  model.make_rtm({{3}});
  model.make_rtm({{4}});
  model.make_rtm({{5}});
  model.make_rtm({{6}});
  // Construct lattices
  // UO2 lattice pins (pg. 7)
  vecvec<int> uo2_rtm_ids = to_vecvec<int>(R"(
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
  vecvec<int> mox_rtm_ids = to_vecvec<int>(R"(
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
  vecvec<int> mod_rtm_ids = to_vecvec<int>(R"(
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
  model.make_lattice(uo2_rtm_ids);
  model.make_lattice(mox_rtm_ids);
  model.make_lattice(mod_rtm_ids);
  // Make assemblies
  size_t const num_axial_segments = 10;
  size_t const num_water_segments = num_axial_segments / 10;
  size_t const num_fuel_segments = num_axial_segments - num_water_segments;
  double const slice_height = model_height / num_axial_segments;
  std::vector<double> axial_slice_heights(num_axial_segments + 1);
  for (size_t i = 0; i < num_axial_segments + 1; ++i) {
    axial_slice_heights[i] = i * slice_height;
  }
  std::vector<int> uo2_lattice_ids(num_fuel_segments, 0);
  uo2_lattice_ids.insert(uo2_lattice_ids.end(), num_water_segments, 2);
  std::vector<int> mox_lattice_ids(num_fuel_segments, 1);
  mox_lattice_ids.insert(mox_lattice_ids.end(), num_water_segments, 2);
  std::vector<int> mod_lattice_ids(num_axial_segments, 2);
  model.make_assembly(uo2_lattice_ids, axial_slice_heights);
  model.make_assembly(mox_lattice_ids, axial_slice_heights);
  model.make_assembly(mod_lattice_ids, axial_slice_heights);
  // Make core
  // Core assembly IDs (pg. 6)
  vecvec<int> core_assembly_ids = to_vecvec<int>(R"(
      0 1 2
      1 0 2
      2 2 2
    )");
  model.make_core(core_assembly_ids);

  // Overlay the spatial partition
  gmsh::model::occ::overlay_spatial_partition(model);
  gmsh::model::mesh::generate(2);
  //    gmsh::fltk::run();
  gmsh::write("c5g7.inp");
  model.import_coarse_cells("c5g7.inp");
  export_mesh("c5g7.xdmf", model);

  finalize();
  return 0;
}
