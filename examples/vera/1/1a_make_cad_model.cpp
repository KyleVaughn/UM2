// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // Parameters
  double const r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
  double const r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg.4)
  std::vector<double> const radii = {r_fuel, r_gap, r_clad};
  double const pitch = 1.26;  // Pitch = 1.26 cm (pg. 4)
  double const x = pitch / 2; // x-coordinate of fuel pin center
  double const y = pitch / 2; // y-coordinate of fuel pin center

  std::vector<um2::Material> const materials = {
      um2::Material("Fuel", "forestgreen"),
      um2::Material("Gap", "white"),
      um2::Material("Clad", "slategray"),
  };

  um2::gmsh::model::occ::addCylindricalPin2D({x, y}, radii, materials);
  // um2::gmsh::fltk::run();
  um2::gmsh::write("1a.brep", /*extra_info=*/true);

  um2::finalize();
  return 0;
}
