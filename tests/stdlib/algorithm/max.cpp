#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/math/abs.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(max_int)
{
  static_assert(um2::max(0, 1) == 1);
  static_assert(um2::max(1, 0) == 1);
  static_assert(um2::max(0, 0) == 0);
}
MAKE_CUDA_KERNEL(max_int);

HOSTDEV
TEST_CASE(max_float)
{
  static_assert(um2::abs(um2::max(0.0F, 1.0F) - 1.0F) < 1e-6F);
  static_assert(um2::abs(um2::max(1.0F, 0.0F) - 1.0F) < 1e-6F);
  static_assert(um2::abs(um2::max(0.0F, 0.0F) - 0.0F) < 1e-6F);
}
MAKE_CUDA_KERNEL(max_float);

TEST_SUITE(max)
{
  TEST_HOSTDEV(max_int);
  TEST_HOSTDEV(max_float);
}

auto
main() -> int
{
  RUN_SUITE(max);
  return 0;
}
