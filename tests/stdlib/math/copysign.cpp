#include <um2/config.hpp>
#include <um2/stdlib/math/copysign.hpp>

#include "../../test_macros.hpp"

// We want true floating point equality here
// NOLINTBEGIN(clang-diagnostic-float-equal) OK
#pragma GCC diagnostic push // OK
#pragma GCC diagnostic ignored "-Wfloat-equal"

HOSTDEV
TEST_CASE(copysign_float)
{
  static_assert(um2::copysign(1.0F, 2.0F) == 1.0F);
  static_assert(um2::copysign(1.0F, -2.0F) == -1.0F);
}
MAKE_CUDA_KERNEL(copysign_float);

HOSTDEV
TEST_CASE(copysign_double)
{
  static_assert(um2::copysign(1.0, 2.0) == 1.0);
  static_assert(um2::copysign(1.0, -2.0) == -1.0);
}
MAKE_CUDA_KERNEL(copysign_double);

#pragma GCC diagnostic pop
// NOLINTEND(clang-diagnostic-float-equal)

TEST_SUITE(copysign_suite)
{
  TEST_HOSTDEV(copysign_float);
  TEST_HOSTDEV(copysign_double);
}

auto
main() -> int
{
  RUN_SUITE(copysign_suite);
  return 0;
}
