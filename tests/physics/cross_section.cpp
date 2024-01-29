#include <um2/physics/cross_section.hpp>

#include "../test_macros.hpp"

TEST_CASE(get1GroupTotalXS)
{
  F constexpr eps = condCast<F>(1e-6);
  um2::CrossSection xsec;
  xsec.t() = {2, 11, 5, 3, 4};
  xsec.validate();
  F const max_1g = xsec.get1GroupTotalXS(um2::XSReductionStrategy::Max);
  ASSERT_NEAR(max_1g, 11, eps);
  F const mean_1g = xsec.get1GroupTotalXS(); // mean by default
  ASSERT_NEAR(mean_1g, 5, eps);
}

TEST_SUITE(CrossSection)
{
  TEST(get1GroupTotalXS);
}

auto
main() -> int
{
  RUN_SUITE(CrossSection);
  return 0;
}
