#include <um2/physics/cross_section_library.hpp>
#include <um2/common/settings.hpp>
#include <um2/common/logger.hpp>
#include <um2/common/cast_if_not.hpp>

#include "../test_macros.hpp"

TEST_CASE(readMPACTLibrary)
{
  auto const eps = castIfNot<Float>(1e-4);
  um2::XSLibrary const lib8(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_8G); 
  ASSERT(lib8.numGroups() == 8);
  ASSERT(lib8.groupBounds().size() == 8);
  ASSERT_NEAR(lib8.groupBounds().front(), castIfNot<Float>(2e7), eps); 
  ASSERT(lib8.chi().size() == 8);
  ASSERT_NEAR(lib8.chi().back(), 0, eps); 
  ASSERT(lib8.nuclides().size() == 295);

  um2::XSLibrary const lib51(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);
  ASSERT(lib51.numGroups() == 51);
  ASSERT(lib51.groupBounds().size() == 51);
  ASSERT_NEAR(lib51.groupBounds().front(), castIfNot<Float>(2e7), eps);
  ASSERT(lib51.chi().size() == 51);
  ASSERT_NEAR(lib51.chi().back(), 0, eps);
  ASSERT(lib51.nuclides().size() == 298);
}

TEST_SUITE(XSLibrary) 
{ 
  TEST(readMPACTLibrary); 
}

auto
main() -> int
{
  um2::logger::level = um2::logger::levels::error;
  RUN_SUITE(XSLibrary);
  return 0;
}
