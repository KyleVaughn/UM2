#include <um2/stdlib/math/roots.hpp>

#include "../../test_macros.hpp"

//=============================================================================
// cbrt
//=============================================================================

HOSTDEV
TEST_CASE(cbrt_float)
{
  ASSERT_NEAR(um2::cbrt(8.0F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::cbrt(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::cbrt(1.0F), 1.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(cbrt_float);

//=============================================================================
// sqrt
//=============================================================================

HOSTDEV
TEST_CASE(sqrt_float)
{
  ASSERT_NEAR(um2::sqrt(4.0F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::sqrt(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::sqrt(1.0F), 1.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(sqrt_float);

TEST_SUITE(cbrt) { TEST_HOSTDEV(cbrt_float); }

TEST_SUITE(sqrt) { TEST_HOSTDEV(sqrt_float); }

auto
main() -> int
{
  RUN_SUITE(cbrt);
  RUN_SUITE(sqrt);
  return 0;
}
