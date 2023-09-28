// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  // Before using any functions in the UM2 C++ API, UM2 must be initialized
  um2::initialize();

  // VERA Problem 1A is a 2D single pin-cell problem. To create the CAD model,
  // we will add geometry for everything except the moderator region, which we will
  // create in a separate step.
  //
  // We can utilize one of UM2's convenience functions to easily create the pin geometry.
  // We will call addCylindricalPin2D, which takes 3 arguments:
  //  - center (um2::Point2d)
  //      The x and y coordinates of the center of the pin
  //  - radii (std::vector<double>)
  //      The radii of the pin regions, from the inside to the outside
  //  - materials (std::vector<um2::Material>)
  //      The materials of the pin regions, from the inside to the outside

  // Parameters for the pin-cell geometry
  double const r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
  double const r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg.4)
  double const pitch = 1.26;    // Pitch = 1.26 cm (pg. 4)

  // Materials for the pin regions require a name and a color.
  // Common colors can be created by name (e.g. "forestgreen"), otherwise all colors
  // can be created by RGB/RGBA values.
  // The color swatches for named colors can be found at
  // http://juliagraphics.github.io/Colors.jl/dev/namedcolors/
  // However, UM2 only supports the subset of colors without numbers in their names.
  // For example, ivory is supported, but ivory1 is not.
  //
  // Materials can be created like so:
  um2::Color const red(1.0, 0.0, 0.0);             // Red using floating point RGB values
  um2::Color const slategray(112, 128, 144);       // Slategray using integer RGB values
  um2::Material const fuel("Fuel", "forestgreen"); // forestgreen using a named color
  um2::Material const gap("Gap", red);
  um2::Material const clad("Clad", slategray);

  // UM2 assumes that the lower left corner of the domain is at (0, 0),
  // therefore we will center the pin at half the pitch in both directions.
  um2::Point2d const center = {pitch / 2, pitch / 2};

  // radii for the pin regions, from the inside to the outside
  std::vector<double> const radii = {r_fuel, r_gap, r_clad};

  // materials for the pin regions, from the inside to the outside
  std::vector<um2::Material> const materials = {fuel, gap, clad};

  // Create the pin geometry
  um2::gmsh::model::occ::addCylindricalPin2D(center, radii, materials);

  // Uncomment the following line to view the model in Gmsh.
  // um2::gmsh::fltk::run();
  //
  // NOTE: if you installed UM2's dependencies using spack and the "server.yaml" spack
  // file, this may not work due to the lack of fltk support.
  //
  // With the Gmsh gui open (using the above line), the model can be viewed.
  // Ctrl+Shift+N will open the options menu.
  // To visualize surfaces:
  //  Geometry->Visibility->Surfaces and Geometry->Aspect->Surface display->Solid
  //  The surfaces are colored by material.
  // Ctrl+Shift+V will open the visibility menu.
  // The visibility of the "physical groups", i.e. labeled groups of surfaces, can be
  // toggled here. The 3 groups are the materials: "Material_Fuel", "Material_Gap",
  // and "Material_Clad".

  // Lastly, we will write the model to a file. Since most CAD software does not properly
  // support material data, but does support color data, if each material has a unique
  // color, we can transfer effectively encode the material data using colors.
  // This is necessary when creating models for UM2 in other CAD software.
  // However, since we are using UM2 to create the model, we can use the "extra_info"
  // option to simply write the group and color of each entity to an additional file.
  //
  // Write the model to a "brep" or "step" file.
  um2::gmsh::write("1a.brep", /*extra_info=*/true);

  // Finalize UM2 to free all memory and write any buffered log messages
  um2::finalize();

  return 0;
}
