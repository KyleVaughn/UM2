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

  //============================================================================
  // Materials
  //============================================================================
  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  // Aluminum
  //---------------------------------------------------------------------------
  um2::Material aluminum;
  aluminum.setName("Aluminum");
  aluminum.setDensity(2.7); // g/cm^3
  aluminum.setTemperature(300.0);
  aluminum.setColor(um2::gray);
  aluminum.addNuclideWt("Al27", 0.9725);
  aluminum.addNuclideWt("Mg00", 0.01);
  aluminum.addNuclideWt("Si00", 0.006);
  aluminum.addNuclideWt("Fe00", 0.0035);
  aluminum.addNuclideWt("Cu63", 0.00205437585615717);
  aluminum.addNuclideWt("Cu65", 0.00094562414384283);
  aluminum.addNuclideWt("Cr00", 0.003);
  // No Zinc
  aluminum.addNuclideWt("Ti00", 0.0005);
  aluminum.addNuclideWt("Mn55", 0.0005);
  aluminum.populateXSec(xslib);

  // Water
  //---------------------------------------------------------------------------
  um2::Material h2o;
  h2o.setName("Water");
  h2o.setDensity(0.99821); // g/cm^3
  h2o.setTemperature(300.0);
  h2o.setColor(um2::blue);
  h2o.addNuclidesAtomPercent({"H1", "O16"}, {2.0 / 3.0, 1.0 / 3.0});
  h2o.populateXSec(xslib);

  // Heavy water
  //---------------------------------------------------------------------------
  um2::Material d2o;
  d2o.setName("HeavyWater");
  d2o.setDensity(1.11); // g/cm^3
  d2o.setTemperature(300.0);
  d2o.setColor(um2::darkblue);
  d2o.addNuclidesAtomPercent({"H1", "H2", "O16"}, {0.005 / 3.0, 1.995 / 3.0, 1.0 / 3.0});
  d2o.populateXSec(xslib);

  // LEU fuel
  //---------------------------------------------------------------------------
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setDensity(2.89924);
  fuel.setTemperature(300.0);
  fuel.setColor(um2::red);
  // Weights from (1), pg. 347
  Float const u235_wt = 140.61;
  Float const u234_wt = 1.51;
  Float const u236_wt = 0.75;
  Float const u238_wt = 8.04;
  Float const aluminum_wt = 908;
  Float const iron_wt = 3.7;
  Float const heu_wt = u235_wt + u234_wt + u236_wt + u238_wt + aluminum_wt + iron_wt;
  fuel.addNuclideWt("U235", u235_wt / heu_wt);
  fuel.addNuclideWt("U234", u234_wt / heu_wt);
  fuel.addNuclideWt("U236", u236_wt / heu_wt);
  fuel.addNuclideWt("U238", u238_wt / heu_wt);
  fuel.addNuclideWt("Al27", 0.9725 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Mg00", 0.01 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Si00", 0.006 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Cu63", 0.00205437585615717 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Cu65", 0.00094562414384283 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Cr00", 0.003 * aluminum_wt / heu_wt);
  // No Zinc
  fuel.addNuclideWt("Ti00", 0.0005 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Mn55", 0.0005 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Fe00", iron_wt / heu_wt + 0.0035 * aluminum_wt / heu_wt);
  fuel.populateXSec(xslib);

  // Borated steel
  //---------------------------------------------------------------------------
  um2::Material borated_steel;
  borated_steel.setName("BoratedSteel");
  borated_steel.setDensity(8.0369);
  borated_steel.setTemperature(300.0);
  borated_steel.setColor(um2::black);
  // From (1), pg. 346
  borated_steel.addNuclide("B10", 0.001108);
  borated_steel.addNuclide("B11", 0.005184);
  borated_steel.addNuclide("Fe00", 0.05644);
  borated_steel.addNuclide("Ni00", 0.0113);
  borated_steel.addNuclide("Cr00", 0.0164);
  borated_steel.populateXSec(xslib);

  // Steel
  //---------------------------------------------------------------------------
  um2::Material steel;
  steel.setName("Steel");
  steel.setDensity(7.85);
  steel.setTemperature(300.0);
  steel.setColor(um2::darkgray);
  // Same as above without boron
  steel.addNuclide("Fe00", 0.05644);
  steel.addNuclide("Ni00", 0.0113);
  steel.addNuclide("Cr00", 0.0164);
  steel.populateXSec(xslib);

  um2::Vector<um2::Material> const materials = {aluminum, h2o,           d2o,
                                                fuel,     borated_steel, steel};

  // Import the FSR mesh
  um2::PolytopeSoup soup(filename);

  // Extract the source
  um2::Vector<um2::Vec2F> const source = um2::mpact::getSource(soup, materials);

  // Compute the scattering and fission source given k_eff
  Float constexpr k_eff = 1.1165101;
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
