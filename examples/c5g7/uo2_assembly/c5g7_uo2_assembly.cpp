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
  gmsh::open("c5g7_uo2_assembly.brep", true);

  // Construct the MPACT spatial partition using pin-modular ray tracing.
  double const pin_pitch = 1.26; // Pitch = 1.26 cm (pg. 3)
  //    double const assembly_height = 192.78;    // Assembly height = 192.78 cm (pg. 3)
  double const model_height = 214.20;     // Model height = 214.20 cm (pg. 3)
  Vec2d const dxdy(pin_pitch, pin_pitch); // Pin pitch in x and y directions
  mpact::SpatialPartition<double, int32_t> model;
  // Construct coarse cells
  // 0 UO2
  // 1 Fission Chamber
  // 2 Guide Tube
  // 3 Moderator
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  model.make_coarse_cell(dxdy);
  // Construct RTMs (pin-modular ray tracing)
  model.make_rtm({{0}});
  model.make_rtm({{1}});
  model.make_rtm({{2}});
  model.make_rtm({{3}});
  // Construct lattices
  // UO2 lattice pins (pg. 7)
  vecvec<int> uo2_rtm_ids = to_vecvec<int>(R"(
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 2 0 0 2 0 0 2 0 0 0 0 0
      0 0 0 2 0 0 0 0 0 0 0 0 0 2 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 2 0 0 2 0 0 1 0 0 2 0 0 2 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 2 0 0 0 0 0 0 0 0 0 2 0 0 0
      0 0 0 0 0 2 0 0 2 0 0 2 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    )");
  // Moderator lattice
  vecvec<int> mod_rtm_ids = to_vecvec<int>(R"(
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
    )");
  model.make_lattice(uo2_rtm_ids);
  model.make_lattice(mod_rtm_ids);
  // Make assemblies
  size_t const num_axial_segments = 10; // Must be divisible by 10
  size_t const num_water_segments = num_axial_segments / 10;
  size_t const num_fuel_segments = num_axial_segments - num_water_segments;
  double const slice_height = model_height / num_axial_segments;
  std::vector<double> axial_slice_heights(num_axial_segments + 1);
  for (size_t i = 0; i < num_axial_segments + 1; ++i) {
    axial_slice_heights[i] = i * slice_height;
  }
  std::vector<int> uo2_lattice_ids(num_fuel_segments, 0);
  uo2_lattice_ids.insert(uo2_lattice_ids.end(), num_water_segments, 1);
  model.make_assembly(uo2_lattice_ids, axial_slice_heights);
  model.make_core({{0}});

  gmsh::model::occ::overlay_spatial_partition(model);
  gmsh::model::mesh::generate(2);
  //    gmsh::fltk::run();
  gmsh::write("c5g7_uo2_assembly.inp");
  model.import_coarse_cells("c5g7_uo2_assembly.inp");
  export_mesh("c5g7_uo2_assembly.xdmf", model);

  finalize();
  return 0;
}
