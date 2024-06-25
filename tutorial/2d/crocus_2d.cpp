// Model reference:
//  Paratte, J. M., et al. "A benchmark on the calculation of kinetic parameters based
//  on reactivity effect experiments in the CROCUS reactor." Annals of Nuclear energy
//  33.8 (2006): 739-748.
//  https://doi.org/10.1016/j.anucene.2005.09.012

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // Parameters
  double const d_uo2_fuel = 1.052;    // UO2 fuel diameter pg. 741 Fig. 2
  double const d_uo2_clad = 1.260;    // UO2 clad diameter pg. 741 Fig. 2
  double const t_uo2_clad = 0.085;    // UO2 clad thickness pg. 741 Fig. 2
  double const uo2_pitch = 1.837;     // UO2 pin pitch pg. 742 Sec. 2.3
  double const d_umetal_fuel = 1.700; // Umetal fuel diameter pg. 741 Fig. 2
  double const d_umetal_clad = 1.935; // Umetal clad diameter pg. 741 Fig. 2
  double const t_umetal_clad = 0.100; // Umetal clad thickness pg. 741 Fig. 2
  double const umetal_pitch = 2.917;  // Umetal pin pitch pg. 742 Sec. 2.3

  // Computed parameters
  double const r_uo2_fuel = d_uo2_fuel / 2;
  double const r_uo2_clad = d_uo2_clad / 2;
  double const r_uo2_gap = r_uo2_clad - t_uo2_clad;
  double const r_umetal_fuel = d_umetal_fuel / 2;
  double const r_umetal_clad = d_umetal_clad / 2;
  double const r_umetal_gap = r_umetal_clad - t_umetal_clad;

  std::vector<double> const uo2_radii = {r_uo2_fuel, r_uo2_gap, r_uo2_clad};
  std::vector<double> const umetal_radii = {r_umetal_fuel, r_umetal_gap, r_umetal_clad};

  // Materials
  um2::Material const uo2("UO2", "forestgreen");
  um2::Material const umetal("Umetal", "yellow");
  um2::Material const gap("Gap", "red");
  um2::Material const clad("Clad", "slategray");

  std::vector<um2::Material> const uo2_mats = {uo2, gap, clad};
  std::vector<um2::Material> const umetal_mats = {umetal, gap, clad};

  // We can efficiently create the model by using the UM2 addCylindricalPinLattice2D
  // function. We will create a lattice for the UO2 pins and a lattice for the Umetal
  // pins. However, we will primarily use NULL pins for blank regions in the lattice.

  std::vector<std::vector<int>> const uo2_pin_ids = um2::to_vecvec<int>(R"(
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0)");

  std::vector<std::vector<int>> const umetal_pin_ids = um2::to_vecvec<int>(R"(
    0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0
    0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0
    0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0
    0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0
    1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0
    1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0
    0 0 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0
    0 0 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0
    0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0)");

  // Depending on how much water we want to simulate, we can change where the center of
  // the problem is located. x_center == y_center == center
  double const center = 15 * umetal_pitch;

  // Compute the offset of the UO2 and Umetal lattices from the center of the problem
  auto const n_uo2 = static_cast<double>(uo2_pin_ids.size());
  auto const n_umetal = static_cast<double>(umetal_pin_ids.size());
  double const uo2_offset = center - 0.5 * n_uo2 * uo2_pitch;
  double const umetal_offset = center - 0.5 * n_umetal * umetal_pitch;

  std::vector<um2::Vec2d> const uo2_dxdy(2, {uo2_pitch, uo2_pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      {{}, uo2_radii},         // Radii of the pins
      {{}, uo2_mats},          // Materials of the pins
      uo2_dxdy,                // Pitch of the pins
      uo2_pin_ids,             // Pin IDs
      {uo2_offset, uo2_offset} // Offset of the lattice
  );

  std::vector<um2::Vec2d> const umetal_dxdy(2, {umetal_pitch, umetal_pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      {{}, umetal_radii},            // Radii of the pins
      {{}, umetal_mats},             // Materials of the pins
      umetal_dxdy,                   // Pitch of the pins
      umetal_pin_ids,                // Pin IDs
      {umetal_offset, umetal_offset} // Offset of the lattice
  );

  // Uncomment to visualize the geometry
  // um2::gmsh::fltk::run();

  // For models that do not easily partition into a regular lattice, we can simply
  // declare each coarse cell to be unique and let UM2 take care of the rest.
  // Here we will create a 20 by 20 regular lattice of coarse cells, overlay the
  // grid onto the geometry, and create a fine mesh for each coarse cell.
  //
  // Note: this is too coarse of a grid to be useful for CMFD, but it is useful
  // for demonstrating the capabilities of UM2.
  um2::mpact::SpatialPartition model;
  size_t const num_cells = 20;

  // Size of the coarse cells
  double const dx = 2.0 * center / static_cast<double>(num_cells);
  um2::Vec2d const dxdy(dx, dx);

  std::vector<std::vector<int>> cc_ids(num_cells, std::vector<int>(num_cells, 0));
  for (size_t i = 0; i < num_cells; ++i) {
    for (size_t j = 0; j < num_cells; ++j) {
      cc_ids[i][j] = static_cast<int>(i * num_cells + j);
      model.makeCoarseCell(dxdy);
      model.makeRTM({{cc_ids[i][j]}});
    }
  }
  model.makeLattice(cc_ids);
  model.makeAssembly({0});
  model.makeCore({{0}});

  // Overlay the spatial partition
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  // Uncomment to visualize the geometry after the spatial partition has been overlaid
  // um2::gmsh::fltk::run();

  um2::gmsh::model::mesh::setGlobalMeshSize(0.2);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);

  um2::gmsh::write("crocus.inp");
  model.importCoarseCells("crocus.inp");
  um2::exportMesh("crocus.xdmf", model);
  um2::finalize();
  return 0;
}
