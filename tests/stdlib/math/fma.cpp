#include <um2/config.hpp>
#include <um2/stdlib/math/fma.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(fma_float)
{
  float const a = 3.0F;
  float const b = 4.0F;
  float const c = 5.0F;
  float const result = um2::fma(a, b, c);
  ASSERT_NEAR(result, 17.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(fma_float);

HOSTDEV
TEST_CASE(fma_double)
{
  double const a = 3.0;
  double const b = 4.0;
  double const c = 5.0;
  double const result = um2::fma(a, b, c);
  ASSERT_NEAR(result, 17.0, 1e-6);
}

TEST_SUITE(fma_suite)
{
  TEST_HOSTDEV(fma_float);
  TEST_HOSTDEV(fma_double);
}

auto
main() -> int
{
  RUN_SUITE(fma_suite);
  return 0;
}
