#include <um2/physics/cross_section.hpp>

#include "../test_macros.hpp"

TEST_CASE(get1GroupTotalXSec)
{
  F constexpr eps = condCast<F>(1e-6);
  um2::XSec xsec;
  xsec.t() = {2, 11, 5, 3, 4};
  xsec.validate();
  F const max_1g = xsec.get1GroupTotalXSec(um2::XSecReduction::Max);
  ASSERT_NEAR(max_1g, 11, eps);
  F const mean_1g = xsec.get1GroupTotalXSec(); // mean by default
  ASSERT_NEAR(mean_1g, 5, eps);
}

TEST_SUITE(XSec) { TEST(get1GroupTotalXSec); }

auto
main() -> int
{
  RUN_SUITE(XSec);
  return 0;
}
