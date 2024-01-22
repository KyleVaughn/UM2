#include <um2/physics/cross_section.hpp>

#include "../test_macros.hpp"

TEST_CASE(getOneGroupTotalXS)
{
#if UM2_ENABLE_FLOAT64
  F const eps = 1e-6;
#else
  F const eps = 1e-6f;
#endif
  um2::CrossSection const xsec({2, 11, 5, 3, 4});
  F const max_1g = xsec.getOneGroupTotalXS(um2::XSReductionStrategy::Max);
  ASSERT_NEAR(max_1g, 11, eps);
  F const mean_1g = xsec.getOneGroupTotalXS(); // mean by default
  ASSERT_NEAR(mean_1g, 5, eps);
}

TEST_SUITE(CrossSection)
{
  TEST(getOneGroupTotalXS);
}

auto
main() -> int
{
  RUN_SUITE(CrossSection);
  return 0;
}
