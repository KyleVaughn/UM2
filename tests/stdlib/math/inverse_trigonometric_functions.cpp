#include <um2/config.hpp>
#include <um2/stdlib/math/inverse_trigonometric_functions.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(acos_float)
{
  ASSERT_NEAR(um2::acos(0.0F), 1.5707963267948966F, 1e-6F); // pi/2
  ASSERT_NEAR(um2::acos(0.5F), 1.0471975511965976F, 1e-6F); // pi/3
  ASSERT_NEAR(um2::acos(1.0F), 0.0F, 1e-6F);                // 0
}
MAKE_CUDA_KERNEL(acos_float);

HOSTDEV
TEST_CASE(asin_float)
{
  ASSERT_NEAR(um2::asin(0.0F), 0.0F, 1e-6F);                // 0
  ASSERT_NEAR(um2::asin(0.5F), 0.5235987755982988F, 1e-6F); // pi/6
  ASSERT_NEAR(um2::asin(1.0F), 1.5707963267948966F, 1e-6F); // pi/2
}

HOSTDEV
TEST_CASE(atan_float)
{
  ASSERT_NEAR(um2::atan(0.0F), 0.0F, 1e-6F);                  // 0
  ASSERT_NEAR(um2::atan(1.0F), 0.7853981633974483F, 1e-6F);   // pi/4
  ASSERT_NEAR(um2::atan(-1.0F), -0.7853981633974483F, 1e-6F); // -pi/4
}

TEST_SUITE(acos_suite) { TEST_HOSTDEV(acos_float); }
TEST_SUITE(asin_suite) { TEST_HOSTDEV(asin_float); }
TEST_SUITE(atan_suite) { TEST_HOSTDEV(atan_float); }

auto
main() -> int
{
  RUN_SUITE(acos_suite);
  RUN_SUITE(asin_suite);
  RUN_SUITE(atan_suite);
  return 0;
}
