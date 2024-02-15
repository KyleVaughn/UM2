#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/math/exponential_functions.hpp>
#include <um2/stdlib/numbers.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(exp_float)
{
  ASSERT_NEAR(um2::exp(0.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::exp(1.0F), um2::e<float>, 1e-6F);
}
MAKE_CUDA_KERNEL(exp_float);

TEST_SUITE(exponential_functions) { TEST_HOSTDEV(exp_float); }

auto
main() -> int
{
  RUN_SUITE(exponential_functions);
  return 0;
}
