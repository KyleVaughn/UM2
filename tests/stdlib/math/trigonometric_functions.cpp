#include <um2/config.hpp>
#include <um2/stdlib/math/trigonometric_functions.hpp>
#include <um2/stdlib/numbers.hpp>

#include "../../test_macros.hpp"

//=============================================================================
// cos
//=============================================================================

HOSTDEV
TEST_CASE(cos_float)
{
  ASSERT_NEAR(um2::cos(0.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::cos(um2::pi_2<float>), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::cos(um2::pi<float>), -1.0F, 1e-6F);
}

//=============================================================================
// sin
//=============================================================================

HOSTDEV
TEST_CASE(sin_float)
{
  ASSERT_NEAR(um2::sin(0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::sin(um2::pi_2<float>), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::sin(um2::pi<float>), 0.0F, 1e-6F);
}

MAKE_CUDA_KERNEL(cos_float);
MAKE_CUDA_KERNEL(sin_float);

TEST_SUITE(cos_suite) { TEST_HOSTDEV(cos_float); }
TEST_SUITE(sin_suite) { TEST_HOSTDEV(sin_float); }

auto
main() -> int
{
  RUN_SUITE(cos_suite);
  RUN_SUITE(sin_suite);
  return 0;
}
