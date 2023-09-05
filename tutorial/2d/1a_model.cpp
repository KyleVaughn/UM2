// Model reference:
//    VERA Core Physics Benchmark Progression Problem Specifications
//    Revision 4, August 29, 2014
//    CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  // Before using any functions in the C++ API, UM2 must be initialized
  um2::initialize();

  // VERA Problem 1A is a single pin-cell problem. To create the CAD model,
  // we will create everything but the moderator region, which will be
  // created in a separate step.

  // Store the radii of the fuel, gap, and clad
  double const r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
  double const r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg.4)

  // Create a vector of radii and a vector of materials in inside-out order
  std::vector<double> const radii = {r_fuel, r_gap, r_clad};
  // Materials need a name and a color. The color can be specified by name
  // (e.g. "forestgreen") or by RGB/RGBA values (e.g. {0.13, 0.55, 0.13, 1.0}).
  // The color swatches can be found at
  // http://juliagraphics.github.io/Colors.jl/dev/namedcolors/ However, only colors
  // without numbers in their names are supported. For example, ivory is supported, but
  // ivory1 is not.
  std::vector<um2::Material> const materials = {
      um2::Material("Fuel", "forestgreen"),
      um2::Material("Gap", "pink"),
      um2::Material("Clad", "slategray"),
  };

  // Create a pin with the specified radii and materials centered at half the pitch
  // of the pin-cell. UM2 assumes that the lower left corner of the model is at (0, 0),
  // therefore the center of the pin is at (pitch / 2, pitch / 2).
  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)
  um2::Point2d const center = {pitch / 2, pitch / 2};
  um2::gmsh::model::occ::addCylindricalPin2D(center, radii, materials);

  // Uncomment the following line to view the model in Gmsh. This may not work
  // if you compiled UM2 using the "server.yaml" spack file, which disables fltk.
  // Ctrl+Shift+N will open the options menu. Geometry->Visibility->Surfaces and
  // Geometry->Aspect->Surface display->Solid may be used to display the model as
  // solid surfaces. The surfaces are colored by material.
  // Ctrl+Shift+V will open the visibility menu. The visibility of the "physical groups",
  // or labeled groups of surfaces, can be toggled here. The 3 groups are the materials
  // "Material_Fuel", "Material_Gap", and "Material_Clad".
  // um2::gmsh::fltk::run();

  // Write the model to a "brep" file. A "step" file can also be written.
  // We will use the "extra_info" option to write the groups and colors of entities
  // to the file. This is necessary for transferring materials to the meshing stage.
  um2::gmsh::write("1a.brep", /*extra_info=*/true);

  // Finalize UM2 to free all memory and write any buffered log messages
  um2::finalize();

  return 0;
}
