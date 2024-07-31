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
  um2::Vector<um2::Material> const materials = um2::getC5G7Materials();
  
  // Import the FSR mesh
  um2::PolytopeSoup soup(filename);

  // Extract the source
  um2::Vector<um2::Vec2F> const source = um2::mpact::getSource(soup, materials);

  // Compute the scattering and fission source given k_eff
  Float constexpr k_eff = 1.1864063;
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

  soup.write("c5g7_source.xdmf");
  um2::finalize();
  return 0;
}
// NOLINTEND(misc-include-cleaner)
