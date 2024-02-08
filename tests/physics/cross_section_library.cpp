#include <um2/physics/cross_section_library.hpp>

#include "../test_macros.hpp"

TEST_CASE(MPACT)
{
  um2::XSLibrary const lib8(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_8G); 
  ASSERT(lib8.numGroups() == 8);
  ASSERT(lib8.groupBounds().size() == 8);
  ASSERT_NEAR(lib8.groupBounds().front(), condCast<F>(2e7), condCast<F>(1e-4));
  ASSERT(lib8.chi().size() == 8);
  ASSERT_NEAR(lib8.chi().back(), 0, condCast<F>(1e-4));
  ASSERT(lib8.nuclides().size() == 295);

  um2::XSLibrary const lib51(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);
  ASSERT(lib51.numGroups() == 51);
  ASSERT(lib51.groupBounds().size() == 51);
  ASSERT_NEAR(lib51.groupBounds().front(), condCast<F>(2e7), condCast<F>(1e-4));
  ASSERT(lib51.chi().size() == 51);
  ASSERT_NEAR(lib51.chi().back(), 0, condCast<F>(1e-4));
  ASSERT(lib51.nuclides().size() == 298);
}

TEST_SUITE(XSLibrary) { TEST(MPACT); }

auto
main() -> int
{
  um2::log::level = um2::log::levels::error;
  RUN_SUITE(XSLibrary);
  return 0;
}
