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

  // Fuel_2.110%
  //---------------------------------------------------------------------------
  um2::Material fuel_2110;
  fuel_2110.setName("Fuel_2.11%");
  fuel_2110.setDensity(10.257);    // g/cm^3, Table P4-1 (pg. 50)
  fuel_2110.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  fuel_2110.setColor(um2::red);    // Match Fig. P4-2 (pg. 53)
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  fuel_2110.addNuclide("O16", 4.57591e-02);
  fuel_2110.addNuclide("U234", 4.04814e-06);
  fuel_2110.addNuclide("U235", 4.88801e-04);
  fuel_2110.addNuclide("U236", 2.23756e-06);
  fuel_2110.addNuclide("U238", 2.23844e-02);
  fuel_2110.populateXSec(xslib);

  // Fuel_2.619%
  //---------------------------------------------------------------------------
  um2::Material fuel_2619;
  fuel_2619.setName("Fuel_2.619%");
  fuel_2619.setDensity(10.257);    // g/cm^3, Table P4-1 (pg. 50)
  fuel_2619.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  fuel_2619.setColor(um2::green);  // Match Fig. P4-2 (pg. 53)
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  fuel_2619.addNuclide("O16", 4.57617e-02);
  fuel_2619.addNuclide("U234", 5.09503e-06);
  fuel_2619.addNuclide("U235", 6.06733e-04);
  fuel_2619.addNuclide("U236", 2.76809e-06);
  fuel_2619.addNuclide("U238", 2.22663e-02);
  fuel_2619.populateXSec(xslib);

  // Gap
  //---------------------------------------------------------------------------
  um2::Material gap;
  gap.setName("Gap");
  // He at 565 K and 2250 psia, according to NIST has a density of 0.012768 kg/m^3.
  gap.setDensity(0.00012768); // g/cm^3,
  gap.setTemperature(565.0);  // K, Table P4-1 (pg. 50)
  gap.setColor(um2::yellow);
  gap.addNuclideWt("He4", 1.0);
  gap.populateXSec(xslib);

  // Clad
  //---------------------------------------------------------------------------
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(6.56);      // g/cm^3, (pg. 18)
  clad.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  clad.setColor(um2::slategray);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  clad.addNuclide(24050, 3.30121e-06);
  clad.addNuclide(24052, 6.36606e-05);
  clad.addNuclide(24053, 7.21860e-06);
  clad.addNuclide(24054, 1.79686e-06);
  clad.addNuclide(26054, 8.68307e-06);
  clad.addNuclide(26056, 1.36306e-04);
  clad.addNuclide(26057, 3.14789e-06);
  clad.addNuclide(26058, 4.18926e-07);
  clad.addNuclide(40090, 2.18865e-02);
  clad.addNuclide(40091, 4.77292e-03);
  clad.addNuclide(40092, 7.29551e-03);
  clad.addNuclide(40094, 7.39335e-03);
  clad.addNuclide(40096, 1.19110e-03);
  clad.addNuclide(50112, 4.68066e-06);
  clad.addNuclide(50114, 3.18478e-06);
  clad.addNuclide(50115, 1.64064e-06);
  clad.addNuclide(50116, 7.01616e-05);
  clad.addNuclide(50117, 3.70592e-05);
  clad.addNuclide(50118, 1.16872e-04);
  clad.addNuclide(50119, 4.14504e-05);
  clad.addNuclide(50120, 1.57212e-04);
  clad.addNuclide(50122, 2.23417e-05);
  clad.addNuclide(50124, 2.79392e-05);
  clad.addNuclide(72174, 3.54138e-09);
  clad.addNuclide(72176, 1.16423e-07);
  clad.addNuclide(72177, 4.11686e-07);
  clad.addNuclide(72178, 6.03806e-07);
  clad.addNuclide(72179, 3.01460e-07);
  clad.addNuclide(72180, 7.76449e-07);
  clad.populateXSec(xslib);

  // Moderator
  //---------------------------------------------------------------------------
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743);     // g/cm^3, Table P1-1 (pg. 20)
  moderator.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  moderator.setColor(um2::blue);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  moderator.addNuclide(1001, 4.96194e-02);
  moderator.addNuclide(5010, 1.12012e-05);
  moderator.addNuclide(5011, 4.50862e-05);
  moderator.addNuclide(8016, 2.48097e-02);
  moderator.populateXSec(xslib);

  // Pyrex
  //---------------------------------------------------------------------------
  um2::Material pyrex;
  pyrex.setName("Pyrex");
  pyrex.setDensity(2.25);      // g/cm^3, Table 6 (pg. 8)
  pyrex.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  pyrex.setColor(um2::orange);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 55)
  pyrex.addNuclide(5010, 9.63266e-04);
  pyrex.addNuclide(5011, 3.90172e-03);
  pyrex.addNuclide(8016, 4.67761e-02);
  pyrex.addNuclide(14000, 1.97326e-02); // Natural Silicon
  pyrex.populateXSec(xslib);

  // SS304
  //---------------------------------------------------------------------------
  um2::Material ss304;
  ss304.setName("SS304");
  ss304.setDensity(8.0);       // g/cm^3, (pg. 18)
  ss304.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  ss304.setColor(um2::darkgray);
  ss304.addNuclide(6000, 3.20895e-04);  // Natural Carbon
  ss304.addNuclide(14000, 1.71537e-03); // Natural Silicon
  ss304.addNuclide(15031, 6.99938e-05);
  ss304.addNuclide(24050, 7.64915e-04);
  ss304.addNuclide(24052, 1.47506e-02);
  ss304.addNuclide(24053, 1.67260e-03);
  ss304.addNuclide(24054, 4.16346e-04);
  ss304.addNuclide(25055, 1.75387e-03);
  ss304.addNuclide(26054, 3.44776e-03);
  ss304.addNuclide(26056, 5.41225e-02);
  ss304.addNuclide(26057, 1.24992e-03);
  ss304.addNuclide(26058, 1.66342e-04);
  ss304.addNuclide(28058, 5.30854e-03);
  ss304.addNuclide(28060, 2.04484e-03);
  ss304.addNuclide(28061, 8.88879e-05);
  ss304.addNuclide(28062, 2.83413e-04);
  ss304.addNuclide(28064, 7.21770e-05);
  ss304.populateXSec(xslib);

  // AIC
  //---------------------------------------------------------------------------
  um2::Material aic;
  aic.setName("AIC");
  aic.setDensity(10.2);      // g/cm^3, Table 8 (pg. 10)
  aic.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  aic.setColor(um2::purple);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 55)
  aic.addNuclide(47107, 2.36159e-02);
  aic.addNuclide(47109, 2.19403e-02);
  aic.addNuclide(48000, 2.73220e-03); // Natural Cadmium
  aic.addNuclide(49113, 3.44262e-04);
  aic.addNuclide(49115, 7.68050e-03);
  aic.populateXSec(xslib);

  um2::Vector<um2::Material> const materials = {fuel_2110, fuel_2619, gap,   clad,
                                                moderator, pyrex,     ss304, aic};

  // Import the FSR mesh
  um2::PolytopeSoup soup(filename);

  // Extract the source
  um2::Vector<um2::Vec2F> const source = um2::mpact::getSource(soup, materials);

  // Compute the scattering and fission source given k_eff
  Float constexpr k_eff = 0.9482152;
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
