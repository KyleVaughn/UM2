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
  double assembly_pitch = 21.42;   // assembly_pitch = 21.42 cm (pg. 3)
  double assembly_height = 192.78; // Assembly height = 192.78 cm (pg. 3)

  // Materials
  Material uo2("UO2", "forestgreen");
  Material mox43("MOX_4.3", "orange");
  Material mox70("MOX_7.0", "red3");
  Material mox87("MOX_8.7", "red4");
  Material fiss_chamber("Fission Chamber", "black");
  Material guide_tube("Guide Tube", "darkgrey");

  // Create the lattices using add_cylindrical_pin_lattice
  vecvec<double> pin_radii(6, {radius});
  std::vector<Vec2d> dxdy(6, {pin_pitch, pin_pitch});
  vecvec<Material> pin_mats = {{uo2},   {mox43},        {mox70},
                               {mox87}, {fiss_chamber}, {guide_tube}};

  // UO2 lattice pins (pg. 7)
  vecvec<int> uo2_lattice = to_vecvec<int>(R"(
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
  vecvec<int> mox_lattice = to_vecvec<int>(R"(
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

  namespace factory = gmsh::model::occ;
  // UO2 lattices
  factory::add_cylindrical_pin_lattice(pin_radii, pin_mats, assembly_height, dxdy,
                                       uo2_lattice, assembly_pitch * Point3d(0, 2, 0));
  factory::add_cylindrical_pin_lattice(pin_radii, pin_mats, assembly_height, dxdy,
                                       uo2_lattice, assembly_pitch * Point3d(1, 1, 0));
  // MOX lattices
  factory::add_cylindrical_pin_lattice(pin_radii, pin_mats, assembly_height, dxdy,
                                       mox_lattice, assembly_pitch * Point3d(0, 1, 0));
  factory::add_cylindrical_pin_lattice(pin_radii, pin_mats, assembly_height, dxdy,
                                       mox_lattice, assembly_pitch * Point3d(1, 2, 0));

  //    gmsh::fltk::run();
  gmsh::write("c5g7.brep", true);
  finalize();
  return 0;
}
