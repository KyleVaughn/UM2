#include <iostream>
#include <um2.hpp>

auto
// NOLINTNEXTLINE(bugprone-exception-escape)
main() -> int
{
  um2::initialize();
  // Parameters
  double const center = 42; // cm

  // UO2 params
  //  page 741 Fig. 2 diagram
  double const r_uo2_fuel = 1.052 / 2;
  double const r_uo2_clad = 1.260 / 2;
  double const r_uo2_gap = r_uo2_clad - 0.085;

  // page 742 diagram
  double const pin_uo2_pitch = 1.837;
  double const pin_umetal_pitch = 2.917;

  // offset = center location - 0.5 * num_cells * pitch
  double const uo2_offset = center - 11 * pin_uo2_pitch;

  // page 741 Fig. 2 diagram
  double const r_umetal_fuel = 1.7 / 2;
  double const r_umetal_clad = 1.935 / 2;
  double const r_umetal_gap = r_umetal_clad - 0.001;
  double const umetal_offset = center - 10 * pin_umetal_pitch;

  // Materials
  um2::Material const uo2("UO2", "orange");
  um2::Material const umetal("Umetal", "red");
  um2::Material const clad("clad", "green");
  um2::Material const gap("gap", "slategray");

  std::vector<double> const uo2_radii = {r_uo2_fuel, r_uo2_gap, r_uo2_clad}; // uo2 pin
  std::vector<double> const umetal_radii = {r_umetal_fuel, r_umetal_gap,
                                            r_umetal_clad}; // uo2 pin
  std::vector<um2::Material> const uo2_mats = {uo2, gap, clad};
  std::vector<um2::Material> const umetal_mats = {umetal, gap, clad};

  // make UO2;
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

  // make umetal;
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

  std::vector<um2::Vec2d> const uo2_dxdy(2, {pin_uo2_pitch, pin_uo2_pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D({{}, uo2_radii}, // radii
                                                    {{}, uo2_mats},  // materials
                                                    uo2_dxdy, // dx, dy of each pin cell
                                                    uo2_pin_ids, // pin ids
                                                    {uo2_offset, uo2_offset}
                                                    // offset (account for half gap)
  );

  std::vector<um2::Vec2d> const umetal_dxdy(2, {pin_umetal_pitch, pin_umetal_pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      {{}, umetal_radii},            // radii
      {{}, umetal_mats},             // materials
      umetal_dxdy,                   // dx, dy of each pin cell
      umetal_pin_ids,                // pin ids
      {umetal_offset, umetal_offset} // offset (account for half gap)
  );

  // um2::gmsh::fltk::run();

  um2::gmsh::write("crocus.brep", /*extra_info=*/true);

  // mesh
  um2::mpact::SpatialPartition model;
  size_t const num_cells = 20;
  um2::Vec2d const dxdy = {2 * center / num_cells, 2 * center / num_cells};
  std::vector<std::vector<int>> pin_ids(num_cells, std::vector<int>(num_cells, 0));
  for (size_t i = 0; i < num_cells; ++i) {
    for (size_t j = 0; j < num_cells; ++j) {
      pin_ids[i][j] = static_cast<int>(i * num_cells + j);
      model.makeCoarseCell(dxdy);
      model.makeRTM({{pin_ids[i][j]}});
    }
  }
  model.makeLattice(pin_ids);
  model.makeAssembly({0});
  model.makeCore({{0}});

  // Overlay the spatial partition
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  //   um2::gmsh::fltk::run();

  double const lc = 0.15;
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);
  um2::gmsh::model::mesh::generate(2);

  //   um2::gmsh::fltk::run();

  um2::gmsh::write("crocus.inp");
  model.importCoarseCells("crocus.inp");
  um2::exportMesh("crocus.xdmf", model);

  um2::finalize();
  return 0;
}