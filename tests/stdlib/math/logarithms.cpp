#include <um2/stdlib/math/logarithms.hpp>

#include <um2/stdlib/numbers.hpp>

#include "../../test_macros.hpp"

//=============================================================================
// log
//=============================================================================

HOSTDEV
TEST_CASE(log_float)
{
  ASSERT_NEAR(um2::log(1.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::log(um2::e<float>), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::log(um2::e<float> * um2::e<float>), 2.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(log_float);

TEST_SUITE(log) { TEST_HOSTDEV(log_float); }

auto
main() -> int
{
  RUN_SUITE(log);
  return 0;
}
