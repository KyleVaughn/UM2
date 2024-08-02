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



  //============================================================================
  // Materials
  //============================================================================
  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  // NOTE: number densities should be computed from the source, but I have simply
  // ripped them from another CROCUS model for now.

  Float constexpr temp = 293.15; // K. pg. 744 Sec. 3.1

  // UO2
  //---------------------------------------------------------------------------
  um2::Material uo2;
  uo2.setName("UO2");
  uo2.setDensity(10.556); // pg. 742 Sec. 2.3
  uo2.setTemperature(temp);
  uo2.setColor(um2::orange); // Match Fig. 4
  uo2.addNuclide(92235, 4.30565e-04);
  uo2.addNuclide(92238, 2.31145e-02);
  uo2.addNuclide(8016, 4.70902e-02);
  uo2.populateXSec(xslib);

  // Clad
  //---------------------------------------------------------------------------
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(2.70); // pg. 743 Table 1
  clad.setTemperature(temp);
  clad.setColor(um2::slategray);
  clad.addNuclide(13027, 6.02611e-02);
  clad.populateXSec(xslib);

  // Umetal
  //---------------------------------------------------------------------------
  um2::Material umetal;
  umetal.setName("Umetal");
  umetal.setDensity(18.677); // pg. 742 Sec. 2.3
  umetal.setTemperature(temp);
  umetal.setColor(um2::red);
  umetal.addNuclide(92235, 4.53160e-04);
  umetal.addNuclide(92238, 4.68003e-02);
  umetal.populateXSec(xslib);

  // Water
  //---------------------------------------------------------------------------
  um2::Material water;
  water.setName("Water");
  water.setDensity(0.9983); // pg. 743 Table 1
  water.setTemperature(temp);
  water.setColor(um2::blue);
  water.addNuclide(1001, 6.67578e-02);
  water.addNuclide(8016, 3.33789e-02);
  water.populateXSec(xslib);

  um2::Vector<um2::Material> const materials = {uo2, clad, umetal, water};

  // Import the FSR mesh
  um2::PolytopeSoup soup(filename);

  // Extract the source
  um2::Vector<um2::Vec2F> const source = um2::mpact::getSource(soup, materials);

  // Compute the scattering and fission source given k_eff
  Float constexpr k_eff = 1.0305077;
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
