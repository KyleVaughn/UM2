#include <um2/physics/cross_section_library.hpp>

#include "../test_macros.hpp"

TEST_CASE(MPACT)
{
  um2::XSLibrary const lib(XSLIB_PATH);
  //  ASSERT(lib.temperatures().size() == 3);
  //  ASSERT(lib.numGroups() == 8);
}

TEST_SUITE(XSLibrary) { TEST(MPACT); }

auto
main() -> int
{
  RUN_SUITE(XSLibrary);
  return 0;
}
