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

  // Parameters
  double radius = 0.54;            // Pin radius = 0.54 cm (pg. 3)
  double pin_pitch = 1.26;         // pin_pitch = 1.26 cm (pg. 3)
  double assembly_height = 192.78; // Assembly height = 192.78 cm (pg. 3)

  // Materials
  Material uo2("UO2", "forestgreen");
  Material fiss_chamber("Fission Chamber", "black");
  Material guide_tube("Guide Tube", "darkgrey");

  vecvec<double> pin_radii(3, {radius});
  std::vector<Vec2d> dxdy(3, {pin_pitch, pin_pitch});
  vecvec<Material> pin_mats = {{uo2}, {fiss_chamber}, {guide_tube}};

  // UO2 lattice pins (pg. 7)
  vecvec<int> uo2_lattice = to_vecvec<int>(R"(
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

  namespace factory = gmsh::model::occ;
  // UO2 lattices
  factory::add_cylindrical_pin_lattice(pin_radii, pin_mats, assembly_height, dxdy,
                                       uo2_lattice);
  //    gmsh::fltk::run();
  gmsh::write("c5g7_uo2_assembly.brep", true);
  finalize();
  return 0;
}
