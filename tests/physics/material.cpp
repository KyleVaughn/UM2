#include <um2/physics/cross_section.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(getOneGroupTotalXS)
{
  T const eps = static_cast<T>(1e-6);
  um2::CrossSection<T> const xsec({2, 11, 5, 3, 4});
  T const max_1g = xsec.getOneGroupTotalXS(um2::XSReductionStrategy::Max);
  ASSERT_NEAR(max_1g, 11, eps);
  T const mean_1g = xsec.getOneGroupTotalXS(); // mean by default
  ASSERT_NEAR(mean_1g, 5, eps);
}

template <typename T>
TEST_SUITE(cross_section)
{
  TEST((getOneGroupTotalXS<T>));
}

auto
main() -> int
{
  RUN_SUITE(cross_section<float>);
  RUN_SUITE(cross_section<double>);
  return 0;
}
