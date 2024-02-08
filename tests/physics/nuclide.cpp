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

TEST_CASE(interpXS)
{
  F constexpr eps = condCast<F>(1e-4);
  um2::Nuclide nuc;
  nuc.temperatures() = {300, 600, 900};
  um2::XSec xs0;
  xs0.t() = {1, 2, 3};
  um2::XSec xs1;
  xs1.t() = {4, 5, 6};
  um2::XSec xs2;
  xs2.t() = {7, 8, 9};
  nuc.xs() = {xs0, xs1, xs2};
  ASSERT(nuc.temperatures().size() == 3);
  ASSERT(nuc.xs().size() == 3);

  // Below min
  auto xs = nuc.interpXS(100);
  ASSERT_NEAR(xs.t(0), 1, eps);
  ASSERT_NEAR(xs.t(1), 2, eps);
  ASSERT_NEAR(xs.t(2), 3, eps);

  // At min
  xs = nuc.interpXS(300);
  ASSERT_NEAR(xs.t(0), 1, eps);
  ASSERT_NEAR(xs.t(1), 2, eps);
  ASSERT_NEAR(xs.t(2), 3, eps);

  // At mid
  xs = nuc.interpXS(600);
  ASSERT_NEAR(xs.t(0), 4, eps);
  ASSERT_NEAR(xs.t(1), 5, eps);
  ASSERT_NEAR(xs.t(2), 6, eps);

  // At max
  xs = nuc.interpXS(900);
  ASSERT_NEAR(xs.t(0), 7, eps);
  ASSERT_NEAR(xs.t(1), 8, eps);
  ASSERT_NEAR(xs.t(2), 9, eps);

  // Above max
  xs = nuc.interpXS(1000);
  ASSERT_NEAR(xs.t(0), 7, eps);
  ASSERT_NEAR(xs.t(1), 8, eps);
  ASSERT_NEAR(xs.t(2), 9, eps);

  // Linear interpolation over sqrt of temperature
  F constexpr temp = 450;
  F constexpr v0 = condCast<F>(2.62774);
  xs = nuc.interpXS(temp);
  ASSERT_NEAR(xs.t(0), v0, eps);
  ASSERT_NEAR(xs.t(1), v0 + 1, eps);
  ASSERT_NEAR(xs.t(2), v0 + 2, eps);
}

TEST_SUITE(Nuclide)
{ 
  TEST(toZAID); 
  TEST(interpXS);
}

auto
main() -> int
{
  RUN_SUITE(Nuclide);
  return 0;
}
