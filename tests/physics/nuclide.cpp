#include <um2/physics/nuclide.hpp>
#include <um2/physics/cross_section.hpp>
#include <um2/common/cast_if_not.hpp>
#include <um2/config.hpp>

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
  auto constexpr eps = castIfNot<Float>(1e-4);
  um2::Nuclide nuc;
  nuc.temperatures() = {300, 600, 900};
  um2::XSec xs0(3);
  xs0.a() = {1, 2, 3};
  xs0.f() = {1, 2, 3};
  xs0.nuf() = {1, 2, 3};
  xs0.tr() = {1, 2, 3};
  xs0.s() = {1, 2, 3};
  auto & xs0ss = xs0.ss();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      xs0ss(j, i) = j + 1;
    }
  }
  um2::XSec xs1(3);
  xs1.a() = {4, 5, 6};
  xs1.f() = {4, 5, 6};
  xs1.nuf() = {4, 5, 6};
  xs1.tr() = {4, 5, 6};
  xs1.s() = {4, 5, 6};
  auto & xs1ss = xs1.ss();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      xs1ss(j, i) = j + 4;
    }
  }
  um2::XSec xs2(3);
  xs2.a() = {7, 8, 9};
  xs2.f() = {7, 8, 9};
  xs2.nuf() = {7, 8, 9};
  xs2.tr() = {7, 8, 9};
  xs2.s() = {7, 8, 9};
  auto & xs2ss = xs2.ss();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      xs2ss(j, i) = j + 7;
    }
  }
  nuc.xs() = {xs0, xs1, xs2};
  ASSERT(nuc.temperatures().size() == 3);
  ASSERT(nuc.xs().size() == 3);

  // Below min
  auto xs = nuc.interpXS(100);
  ASSERT_NEAR(xs.a()[0], 1, eps);
  ASSERT_NEAR(xs.a()[1], 2, eps);
  ASSERT_NEAR(xs.a()[2], 3, eps);

  // At min
  xs = nuc.interpXS(300);
  ASSERT_NEAR(xs.f()[0], 1, eps);
  ASSERT_NEAR(xs.f()[1], 2, eps);
  ASSERT_NEAR(xs.f()[2], 3, eps);

  // At mid
  xs = nuc.interpXS(600);
  ASSERT_NEAR(xs.nuf()[0], 4, eps);
  ASSERT_NEAR(xs.nuf()[1], 5, eps);
  ASSERT_NEAR(xs.nuf()[2], 6, eps);

  // At max
  xs = nuc.interpXS(900);
  ASSERT_NEAR(xs.tr()[0], 7, eps);
  ASSERT_NEAR(xs.tr()[1], 8, eps);
  ASSERT_NEAR(xs.tr()[2], 9, eps);

  // Above max
  xs = nuc.interpXS(1000);
  ASSERT_NEAR(xs.ss()(0), 7, eps);
  ASSERT_NEAR(xs.ss()(1), 8, eps);
  ASSERT_NEAR(xs.ss()(2), 9, eps);

  // Linear interpolation over sqrt of temperature
  Float constexpr temp = 450;
  auto constexpr v0 = castIfNot<Float>(2.62774);
  xs = nuc.interpXS(temp);
  ASSERT_NEAR(xs.f()[0], v0, eps);
  ASSERT_NEAR(xs.f()[1], v0 + 1, eps);
  ASSERT_NEAR(xs.f()[2], v0 + 2, eps);
  ASSERT_NEAR(xs.ss()(0), v0, eps);
  ASSERT_NEAR(xs.ss()(1), v0 + 1, eps);
  ASSERT_NEAR(xs.ss()(2), v0 + 2, eps);
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
