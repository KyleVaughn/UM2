#include <um2/physics/cross_section_library.hpp>
#include <um2/common/settings.hpp>
#include <um2/common/logger.hpp>
#include <um2/common/cast_if_not.hpp>

#include "../test_macros.hpp"

TEST_CASE(MPACT)
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

TEST_CASE(getXS)
{
  um2::logger::level = um2::logger::levels::info;
  um2::XSLibrary const lib8(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_8G);
  um2::Material fuel;    
  fuel.setName("Fuel");    
  fuel.setDensity(castIfNot<Float>(10.42)); // g/cm^3, Table P1-2 (pg. 20)    
  fuel.setTemperature(castIfNot<Float>(565.0)); // K, Table P1-1 (pg. 20)    
  fuel.setColor(um2::forestgreen);    
  fuel.addNuclide("U235", castIfNot<Float>(1.0));    
  fuel.addNuclide("O16", castIfNot<Float>(1.0));

  auto const xs = lib8.getXS(fuel);
  ASSERT_NEAR(xs.t(0), 9, 1)
  ASSERT_NEAR(xs.t(1), 14, 1)
}

TEST_SUITE(XSLibrary) 
{ 
  TEST(MPACT); 
  TEST(getXS);
}

auto
main() -> int
{
  um2::logger::level = um2::logger::levels::error;
  RUN_SUITE(XSLibrary);
  return 0;
}
