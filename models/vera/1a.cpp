// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  //============================================================================
  // Materials
  //============================================================================
  // Nuclides and number densities from table P1-4 (pg. 23)

  // Fuel
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setDensity(10.42); // g/cm^3, Table P1-2 (pg. 20)
  fuel.setTemperature(565.0); // K, Table P1-1 (pg. 20)
  fuel.setColor(um2::forestgreen);
  fuel.addNuclide("U234", 6.11864e-6); // Number density in atoms/b-cm
  fuel.addNuclide("U235", 7.18132e-4);
  fuel.addNuclide("U236", 3.29861e-6);
  fuel.addNuclide("U238", 2.21546e-2);
  fuel.addNuclide("O16", 4.57642e-2);

//  // Gap
//  um2::Material gap;
//  gap.setName("Gap");
//  // "Helium with nominal density" (pg. 22). He at 565 K and 2250 psia, according to NIST
//  // has a density of 0.012768 g/cm^3.
//  gap.setDensity(0.012768); // g/cm^3,
//  gap.setTemperature(565.0); // K, Hot zero power temperature (pg. 20)
//  gap.setColor(um2::red);
//  gap.addNuclide("He4", 2.68714e-5);

  // Clad
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(6.56); // g/cm^3, (pg. 18)
  clad.setTemperature(565.0); // K, Hot zero power temperature (pg. 20)
  clad.setColor(um2::slategray);
  clad.addNuclide("Zr90", 2.18865e-02);
  clad.addNuclide("Zr91", 4.77292e-03);
  clad.addNuclide("Zr92", 7.29551e-03);
  clad.addNuclide("Zr94", 7.39335e-03);
  clad.addNuclide("Zr96", 1.19110e-03);
  clad.addNuclide("Sn112", 4.68066e-06);
  clad.addNuclide("Sn114", 3.18478e-06);
  clad.addNuclide("Sn115", 1.64064e-06);
  clad.addNuclide("Sn116", 7.01616e-05);
  clad.addNuclide("Sn117", 3.70592e-05);
  clad.addNuclide("Sn118", 1.16872e-04);
  clad.addNuclide("Sn119", 4.14504e-05);
  clad.addNuclide("Sn120", 1.57212e-04);
  clad.addNuclide("Sn122", 2.23417e-05);
  clad.addNuclide("Sn124", 2.79392e-05);
  clad.addNuclide("Fe54", 8.68307e-06);
  clad.addNuclide("Fe56", 1.36306e-04);
  clad.addNuclide("Fe57", 3.14789e-06);
  clad.addNuclide("Fe58", 4.18926e-07);
  clad.addNuclide("Cr50", 3.30121e-06);
  clad.addNuclide("Cr52", 6.36606e-05);
  clad.addNuclide("Cr53", 7.21860e-06);
  clad.addNuclide("Cr54", 1.79686e-06);
  clad.addNuclide("Hf174", 3.54138e-09);
  clad.addNuclide("Hf176", 1.16423e-07);
  clad.addNuclide("Hf177", 4.11686e-07);
  clad.addNuclide("Hf178", 6.03806e-07);
  clad.addNuclide("Hf179", 3.01460e-07);
  clad.addNuclide("Hf180", 7.76449e-07);

  // Moderator
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743); // g/cm^3, Table P1-1 (pg. 20)
  moderator.setTemperature(565.0); // K, Table P1-1 (pg. 20)
  moderator.setColor(um2::blue);
  moderator.addNuclide("O16", 2.48112e-02);
  moderator.addNuclide("H1", 4.96224e-02);
  moderator.addNuclide("B10", 1.07070e-05);
  moderator.addNuclide("B11", 4.30971e-05);

  //============================================================================
  // Geometry
  //============================================================================

  // Parameters for the pin-cell geometry
  double const r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
//  double const r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg.4)
  double const pitch = 1.26;    // Pitch = 1.26 cm (pg. 4)

  um2::Point2 const center = {pitch / 2, pitch / 2};

//  um2::Vector<double> const radii = {r_fuel, r_gap, r_clad};
 um2::Vector<double> const radii = {r_fuel, r_clad};
//  um2::Vector<um2::Material> const materials = {fuel, gap, clad};
  um2::Vector<um2::Material> const materials = {fuel, clad};

  um2::gmsh::model::occ::addCylindricalPin2D(center, radii, materials);

  um2::mpact::SpatialPartition model;
  model.addCoarseCell({pitch, pitch});
  model.fillHierarchy();
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  double const target_kn = 12.0;
  std::vector<um2::Material> const mat = {fuel, clad, moderator};
  um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(2, mat, target_kn);
//  // um2::gmsh::model::mesh::setGlobalMeshSize(0.06);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::fltk::run();
//
//  um2::gmsh::write("c5g7.inp");
//  model.importCoarseCells("c5g7.inp");
//  for (auto const & cc : model.coarse_cells) {
//    um2::Log::info("CC has " + um2::toString(cc.numFaces()) + " faces");
//  }
//  model.materials[6].xs.t = uo2_xs;
//  model.materials[2].xs.t = mox43_xs;
//  model.materials[3].xs.t = mox70_xs;
//  model.materials[4].xs.t = mox87_xs;
//  model.materials[0].xs.t = fiss_chamber_xs;
//  model.materials[1].xs.t = guide_tube_xs;
//  model.materials[5].xs.t = moderator_xs;
  // um2::PolytopeSoup<double, int> soup;
  // model.toPolytopeSoup(soup, /*write_kn=*/true);
  // soup.write("c5g7.xdmf");
//  model.write("c5g7.xdmf", /*write_kn=*/true);
  um2::finalize();
  return 0;
}
