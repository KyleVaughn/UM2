#include <um2/physics/material.hpp>

#include "../test_macros.hpp"

TEST_CASE(addNuclide)
{
  um2::Material m;
  m.addNuclide("H1", 1.0);
  m.addNuclide("He4", 1.0);
  m.addNuclide("U235", 1.0);
  m.addNuclide("U-238", 1.0);
  m.addNuclide("Cm-244", 1.0);
}

TEST_SUITE(Material) { TEST(addNuclide); }

auto
main() -> int
{
  RUN_SUITE(Material);
  return 0;
}
