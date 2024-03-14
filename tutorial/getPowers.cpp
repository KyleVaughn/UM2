#include <um2/mesh/polytope_soup.hpp>
#include <um2/common/logger.hpp>

#include <cstdlib>
#include <iostream>

auto
main(int argc, char **argv) -> int 
{

  if (argc != 2) {
    std::cerr << "Usage: getPowers <file_name>" << std::endl;
    exit(1);
  }

  um2::String const filename(argv[1]);
  um2::PolytopeSoup soup(filename);
  um2::logger::info("Sorting mesh by Morton Code to speed up spatial queries");
  soup.mortonSort();
  auto const subset_pc = um2::getPowerRegions(soup);
  std::cout << "Power, x, y, z" << std::endl;
  for (auto const pc : subset_pc) {
    std::cout << pc.first << ", " << pc.second[0] << ", " << pc.second[1] <<  ", " << pc.second[2] << std::endl;
  }
  return 0;
}
