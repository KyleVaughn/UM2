#include <um2/stdlib/math/inverse_hyperbolic_functions.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(atanh_float)
{
  ASSERT_NEAR(um2::atanh(0.5F), 0.549306F, 1e-3F);
  ASSERT_NEAR(um2::atanh(0.0F), 0.0F, 1e-3F);
}
MAKE_CUDA_KERNEL(atanh_float);

TEST_SUITE(atanh) { TEST_HOSTDEV(atanh_float); }

auto
main() -> int
{
  RUN_SUITE(atanh);
  return 0;
}
