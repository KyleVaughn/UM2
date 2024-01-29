#include <um2/physics/nuclide.hpp>

#include "../test_macros.hpp"

TEST_CASE(toZAID)
{
  ASSERT(um2::toZAID("H1") == 1001);
  ASSERT(um2::toZAID("H2") == 1002);
  ASSERT(um2::toZAID("He4") == 2004);
  ASSERT(um2::toZAID("He-6") == 2006);
  ASSERT(um2::toZAID("Co-60") == 27060);
  ASSERT(um2::toZAID("Cm-244") == 96244);
  ASSERT(um2::toZAID("Cm244") == 96244);
  ASSERT(um2::toZAID("U-235") == 92235);
  ASSERT(um2::toZAID("U235") == 92235);
}

TEST_SUITE(Nuclide)
{
  TEST(toZAID);
}

auto
main() -> int
{
  RUN_SUITE(Nuclide);
  return 0;
}
