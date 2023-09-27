#include <iostream>
#include <um2.hpp>

auto
// NOLINTNEXTLINE(bugprone-exception-escape)
main() -> int
{
  um2::initialize();

  // UO2 params
  //  page 741 Fig. 2 diagram
  double const r_uo2_fuel = 1.052 / 2;
  double const r_uo2_clad = 1.260 / 2;
  double const r_uo2_gap = r_uo2_clad - 0.085;

  // page 742 diagram
  double const pin_uo2_pitch = 1.837;    // cm
  double const pin_umetal_pitch = 2.917; // cm

  // problem center
  double const center = 20 * pin_umetal_pitch / 2; // umetal is 20x20 in our model

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
  // size_t const num_cells = 20;
  //   um2::Vec2d const dxdy = {2 * center / num_cells, 2 * center / num_cells};
  um2::Vec2d const dxdy = {pin_umetal_pitch, pin_umetal_pitch};
  std::vector<std::vector<int>> pin_ids = um2::to_vecvec<int>(R"(
  0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0
  0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
  0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
  0 0 1 1 1 1 1 1 142 143 144 145 1 1 1 1 1 1 0 0
  0 1 1 1 1 1 134 135 136 137 138 139 140 141 1 1 1 1 0 0
  0 1 1 1 1 1 126 127 128 129 130 131 132 133 1 1 1 1 1 0
  1 1 1 1 114 115 116 117 118 119 120 121 122 123 124 125 1 1 1 0
  1 1 1 1 102 103 104 105 106 107 108 109 110 111 112 113 1 1 1 1
  1 1 1 88 89 90 91 92 93 94 95 96 97 98 99 100 101 1 1 1
  1 1 1 74 75 76 77 78 79 80 81 82 83 84 85 86 87 1 1 1
  1 1 1 60 61 62 63 64 65 66 67 68 69 70 71 72 73 1 1 1
  1 1 1 46 47 48 49 50 51 52 53 54 55 56 57 58 59 1 1 1
  1 1 1 1 34 35 36 37 38 39 40 41 42 43 44 45 1 1 1 1
  0 1 1 1 22 23 24 25 26 27 28 29 30 31 32 33 1 1 1 1
  0 1 1 1 1 1 14 15 16 17 18 19 20 21 1 1 1 1 1 0
  0 0 1 1 1 1 6 7 8 9 10 11 12 13 1 1 1 1 1 0
  0 0 1 1 1 1 1 1 2 3 4 5 1 1 1 1 1 1 0 0
  0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
  0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
  0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0)");
  for (int i = 0; i <= 145; ++i) {
    model.makeCoarseCell(dxdy);
    model.makeRTM({{i}});
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