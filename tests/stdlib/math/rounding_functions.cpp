#include <um2/stdlib/math.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(ceil_float)
{
  ASSERT_NEAR(um2::ceil(1.1F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::ceil(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::ceil(-1.1F), -1.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(ceil_float);

HOSTDEV
TEST_CASE(floor_float)
{
  ASSERT_NEAR(um2::floor(1.1F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::floor(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::floor(-1.1F), -2.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(floor_float);

TEST_SUITE(ceil) { TEST_HOSTDEV(ceil_float); }

TEST_SUITE(floor) { TEST_HOSTDEV(floor_float); }

auto
main() -> int
{
  RUN_SUITE(ceil);
  RUN_SUITE(floor);
  return 0;
}
