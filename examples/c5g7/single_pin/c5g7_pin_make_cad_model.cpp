// BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
// TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
// NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // Parameters
  double const radius = 0.54;            // Pin radius = 0.54 cm (pg. 3)
  double const pin_pitch = 1.26;         // pin_pitch = 1.26 cm (pg. 3)
  double const assembly_height = 192.78; // Assembly height = 192.78 cm (pg. 3)

  // Materials
  um2::Material const uo2("UO2", "forestgreen");

  // Create the lattices using add_cylindrical_pin_lattice
  std::vector<double> const radii = {radius};
  um2::Point3d const center(pin_pitch / 2, pin_pitch / 2, 0.0);
  std::vector<um2::Material> const pin_mats = {uo2};

  namespace factory = um2::gmsh::model::occ;
  factory::addCylindricalPin(center, assembly_height, radii, pin_mats);
  um2::gmsh::fltk::run();
  um2::gmsh::write("c5g7_pin.brep", /*extra_info=*/true);
  um2::finalize();
  return 0;
}
