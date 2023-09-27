// BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
// TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
// NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
// NOLINTNEXTLINE(bugprone-exception-escape)
main() -> int
{
  um2::initialize();

  // Parameters
  double const radius = 0.54;            // Pin radius = 0.54 cm (pg. 3)
  double const pin_pitch = 1.26;         // pin_pitch = 1.26 cm (pg. 3)
  double const assembly_pitch = 21.42;   // assembly_pitch = 21.42 cm (pg. 3)

  // Materials
  um2::Material uo2("UO2", "forestgreen");
  um2::Material mox43("MOX_4.3", "orange");
  um2::Material mox70("MOX_7.0", "red");
  um2::Material mox87("MOX_8.7", "yellow");
  um2::Material fiss_chamber("Fission Chamber", "black");
  um2::Material guide_tube("Guide Tube", "darkgrey");

  // Create the lattices using add_cylindrical_pin_lattice
  std::vector<std::vector<double>> const pin_radii(6, {radius});
  std::vector<um2::Vec2d> const dxdy(6, {pin_pitch, pin_pitch});
  std::vector<std::vector<um2::Material>> const pin_mats = {{uo2},   {mox43},        {mox70},
                               {mox87}, {fiss_chamber}, {guide_tube}};

  // UO2 lattice pins (pg. 7)
  std::vector<std::vector<int>> const uo2_lattice = um2::to_vecvec<int>(R"(
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
  std::vector<std::vector<int>> const mox_lattice = um2::to_vecvec<int>(R"(
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

  namespace factory = um2::gmsh::model::occ;
  // UO2 lattices
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy,
                                       uo2_lattice, assembly_pitch * um2::Point2d(0, 2));
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy,
                                       uo2_lattice, assembly_pitch * um2::Point2d(1, 1));
  // MOX lattices
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy,
                                       mox_lattice, assembly_pitch * um2::Point2d(0, 1));
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy,
                                       mox_lattice, assembly_pitch * um2::Point2d(1, 2));

  um2::gmsh::fltk::run();
  um2::gmsh::write("c5g7.brep", /*extra_info=*/true);
  um2::finalize();
  return 0;
}
