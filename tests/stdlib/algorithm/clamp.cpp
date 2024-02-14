#include <um2/stdlib/algorithm/clamp.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(clamp_int)
{
  static_assert(um2::clamp(0, 1, 3) == 1);
  static_assert(um2::clamp(1, 1, 3) == 1);
  static_assert(um2::clamp(2, 1, 3) == 2);
  static_assert(um2::clamp(3, 1, 3) == 3);
  static_assert(um2::clamp(4, 1, 3) == 3);
}

HOSTDEV
TEST_CASE(clamp_float)
{
  static_assert(um2::abs(um2::clamp(0.0F, 1.0F, 3.0F) - 1.0F) < 1e-6F);
  static_assert(um2::abs(um2::clamp(1.0F, 1.0F, 3.0F) - 1.0F) < 1e-6F);
  static_assert(um2::abs(um2::clamp(2.0F, 1.0F, 3.0F) - 2.0F) < 1e-6F);
  static_assert(um2::abs(um2::clamp(3.0F, 1.0F, 3.0F) - 3.0F) < 1e-6F);
  static_assert(um2::abs(um2::clamp(4.0F, 1.0F, 3.0F) - 3.0F) < 1e-6F);
}

MAKE_CUDA_KERNEL(clamp_int);
MAKE_CUDA_KERNEL(clamp_float);

TEST_SUITE(clamp)
{
  TEST_HOSTDEV(clamp_int);
  TEST_HOSTDEV(clamp_float);
}

auto
main() -> int
{
  RUN_SUITE(clamp);
  return 0;
}
