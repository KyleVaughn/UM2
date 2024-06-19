#include <um2/config.hpp>
#include <um2/stdlib/math/hyperbolic_functions.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(tanh_float)
{
  ASSERT_NEAR(um2::tanh(0.549306F), 0.5F, 1e-3F);
  ASSERT_NEAR(um2::tanh(0.0F), 0.0F, 1e-3F);
}
MAKE_CUDA_KERNEL(tanh_float);

TEST_SUITE(test_tanh) { TEST_HOSTDEV(tanh_float); }

auto
main() -> int
{
  RUN_SUITE(test_tanh);
  return 0;
}
