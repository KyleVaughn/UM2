#include <um2/config.hpp>
#include <um2/stdlib/math/abs.hpp>

#if !defined(__CUDA_ARCH__) && !UM2_ENABLE_FASTMATH
#  include <limits>
#endif

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(abs_int)
{
  static_assert(um2::abs(-1) == 1);
  static_assert(um2::abs(0) == 0);
  static_assert(um2::abs(1) == 1);
}
MAKE_CUDA_KERNEL(abs_int);

HOSTDEV
TEST_CASE(abs_float)
{
  // Exact equality is important here
#pragma GCC diagnostic push // OK
#pragma GCC diagnostic ignored "-Wfloat-equal"
  static_assert(um2::abs(-1.0F) == 1.0F);
  static_assert(um2::abs(0.0F) == 0.0F);
  static_assert(um2::abs(-0.0F) == 0.0F);
  static_assert(um2::abs(1.0F) == 1.0F);
#if !defined(__CUDA_ARCH__) && !UM2_ENABLE_FASTMATH
  float constexpr inf = std::numeric_limits<float>::infinity();
  static_assert(um2::abs(inf) == inf);
  static_assert(um2::abs(-inf) == inf);
#endif
#pragma GCC diagnostic pop
}
MAKE_CUDA_KERNEL(abs_float);

TEST_SUITE(abs_suite)
{
  TEST_HOSTDEV(abs_int);
  TEST_HOSTDEV(abs_float);
}

auto
main() -> int
{
  RUN_SUITE(abs_suite);
  return 0;
}
