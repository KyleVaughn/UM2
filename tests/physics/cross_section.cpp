#include <um2/common/cast_if_not.hpp>
#include <um2/config.hpp>
#include <um2/math/stats.hpp>
#include <um2/physics/cross_section.hpp>

#include "../test_macros.hpp"

TEST_CASE(collapseTo1GroupAvg)
{
  auto constexpr eps = castIfNot<Float>(1e-6);
  um2::XSec const xsec = um2::getC5G7XSecs()[0];
  xsec.validate();
  auto const oneg = xsec.collapseTo1GroupAvg();
  // a, f, nuf, tr, s, ss
  ASSERT_NEAR(oneg.a()[0], um2::mean(xsec.a().begin(), xsec.a().end()), eps);
  ASSERT_NEAR(oneg.f()[0], um2::mean(xsec.f().begin(), xsec.f().end()), eps);
  ASSERT_NEAR(oneg.nuf()[0], um2::mean(xsec.nuf().begin(), xsec.nuf().end()), eps);
  ASSERT_NEAR(oneg.tr()[0], um2::mean(xsec.tr().begin(), xsec.tr().end()), eps);
  ASSERT_NEAR(oneg.s()[0], um2::mean(xsec.s().begin(), xsec.s().end()), eps);
  ASSERT_NEAR(oneg.ss()(0), um2::mean(xsec.s().begin(), xsec.s().end()), eps);
}

TEST_SUITE(XSec) { TEST(collapseTo1GroupAvg); }

auto
main() -> int
{
  RUN_SUITE(XSec);
  return 0;
}
