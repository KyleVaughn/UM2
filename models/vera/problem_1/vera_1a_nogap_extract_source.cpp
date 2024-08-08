// NOLINTBEGIN(misc-include-cleaner)

#include <um2.hpp>
#include <um2/mpact/source.hpp>
#include <um2/stdlib/numeric/iota.hpp>

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  // Check the number of arguments
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " <FSRmesh>");
    return 1;
  }

  // Get the FSR file name
  um2::String const filename(argv[1]);

  // Get the model's materials
  //============================================================================
  // Materials
  //============================================================================
  // Nuclides and number densities from table P4-3 (pg. 54)

  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  // Fuel
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setDensity(10.257);    // g/cm^3, Table P1-2 (pg. 20)
  fuel.setTemperature(565.0); // K, Table P1-1 (pg. 20)
  fuel.setColor(um2::forestgreen);
  fuel.addNuclide("U234", 6.11864e-6); // Number density in atoms/b-cm
  fuel.addNuclide("U235", 7.18132e-4);
  fuel.addNuclide("U236", 3.29861e-6);
  fuel.addNuclide("U238", 2.21546e-2);
  fuel.addNuclide("O16", 4.57642e-2);
  fuel.populateXSec(xslib);

  // Clad
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(6.56);      // g/cm^3, (pg. 18)
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
  clad.populateXSec(xslib);

  // Moderator
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743);     // g/cm^3, Table P1-1 (pg. 20)
  moderator.setTemperature(565.0); // K, Table P1-1 (pg. 20)
  moderator.setColor(um2::blue);
  moderator.addNuclide("O16", 2.48112e-02);
  moderator.addNuclide("H1", 4.96224e-02);
  moderator.addNuclide("B10", 1.07070e-05);
  moderator.addNuclide("B11", 4.30971e-05);
  moderator.populateXSec(xslib);

  um2::Vector<um2::Material> const materials = {fuel, clad, moderator};

  // Import the FSR mesh
  um2::PolytopeSoup soup(filename);

  // Extract the source
  um2::Vector<um2::Vec2F> const source = um2::mpact::getSource(soup, materials);

  // Compute the scattering and fission source given k_eff
  Float constexpr k_eff = 1.1857199;
  Float constexpr lambda = 1 / k_eff;
  um2::Vector<Float> scattering_source(source.size());
  um2::Vector<Float> fission_source(source.size());
  um2::Vector<Float> total_source(source.size());
  for (Int i = 0; i < source.size(); ++i) {
    scattering_source[i] = source[i][0];
    fission_source[i] = lambda * source[i][1];
    total_source[i] = scattering_source[i] + fission_source[i];
  }

  // Add each of the sources as an elset to the soup
  um2::Vector<Int> ids(source.size());
  um2::iota(ids.begin(), ids.end(), 0);
  soup.addElset("scattering_source", ids, scattering_source);
  soup.addElset("fission_source", ids, fission_source);
  soup.addElset("total_source", ids, total_source);

  soup.write("source.xdmf");
  um2::finalize();
  return 0;
}
// NOLINTEND(misc-include-cleaner)
