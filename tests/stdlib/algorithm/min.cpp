#include <um2/stdlib/algorithm/min.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(min_int)
{
  static_assert(um2::min(0, 1) == 0);
  static_assert(um2::min(1, 0) == 0);
  static_assert(um2::min(0, 0) == 0);
}
MAKE_CUDA_KERNEL(min_int);

HOSTDEV
TEST_CASE(min_float)
{
  STATIC_ASSERT_NEAR(um2::min(0.0F, 1.0F), 0.0F, 1e-6F);
  STATIC_ASSERT_NEAR(um2::min(1.0F, 0.0F), 0.0F, 1e-6F);
  STATIC_ASSERT_NEAR(um2::min(0.0F, 0.0F), 0.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(min_float);

TEST_SUITE(min)
{
  TEST_HOSTDEV(min_int);
  TEST_HOSTDEV(min_float);
}

auto
main() -> int
{
  RUN_SUITE(min);
  return 0;
}
