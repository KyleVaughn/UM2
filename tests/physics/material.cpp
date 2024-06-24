#include <um2/physics/material.hpp>
#include <um2/physics/cross_section_library.hpp>
#include <um2/common/settings.hpp>
#include <um2/common/cast_if_not.hpp>
#include <um2/common/color.hpp>
#include <um2/config.hpp>

#include "../test_macros.hpp"

TEST_CASE(addNuclide)
{
  um2::Material m;
  m.addNuclide("H1", 1.0);
  m.addNuclide("He4", 1.0);
  m.addNuclide("U235", 1.0);
  m.addNuclide("U-238", 1.0);
  m.addNuclide("Cm-244", 1.0);

  Float constexpr h_wt = 1.00783;
  Float constexpr o_wt = 15.9949;
  Float constexpr h2o_wt = 2 * h_wt + o_wt;
  um2::Material h2o;
  h2o.setDensity(0.75);
  h2o.addNuclideWt("H1", 2 * h_wt / h2o_wt);
  h2o.addNuclideWt("O16", o_wt / h2o_wt);
  Float const h_num_density = 0.75 * 2 * 0.602214076 / h2o_wt; 
  Float const o_num_density = h_num_density / 2; 
  ASSERT_NEAR(h2o.numDensity(0), h_num_density, 1e-6);
  ASSERT_NEAR(h2o.numDensity(1), o_num_density, 1e-6);
}

TEST_CASE(getXS)    
{    
  um2::XSLibrary const lib8(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_8G);    
  um2::Material fuel;        
  fuel.setName("Fuel");        
  fuel.setDensity(castIfNot<Float>(10.42));
  fuel.setTemperature(castIfNot<Float>(565.0));
  fuel.setColor(um2::forestgreen);        
  fuel.addNuclide("U235", castIfNot<Float>(1.0));        
  fuel.addNuclide("O16", castIfNot<Float>(1.0));    
    
  fuel.populateXSec(lib8);
  ASSERT_NEAR(fuel.xsec().t(0), 9, 1)    
  ASSERT_NEAR(fuel.xsec().t(1), 14, 1)    
}

TEST_SUITE(Material) 
{ 
  TEST(addNuclide); 
  TEST(getXS);
}

auto
main() -> int
{
  RUN_SUITE(Material);
  return 0;
}
