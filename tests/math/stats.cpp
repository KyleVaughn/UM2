#include <um2/math/stats.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

HOSTDEV
TEST_CASE(mean)
{
  um2::Vector<F> v = {1, 2, 3, 4, 5};
  F const m = um2::mean(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, condCast<F>(3), eps);
}

HOSTDEV
TEST_CASE(median)
{
  um2::Vector<F> v = {1, 2, 3, 4, 5};
  F const m = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, condCast<F>(3), eps);
  v.push_back(6);
  F const m2 = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m2, condCast<F>(3.5), eps);
}

HOSTDEV
TEST_CASE(variance)
{
  um2::Vector<F> v = {1, 2, 3, 4, 5};
  F const m = um2::variance(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, condCast<F>(2.5), eps);
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

#if UM2_USE_CUDA

MAKE_CUDA_KERNEL(mean);
MAKE_CUDA_KERNEL(median);
MAKE_CUDA_KERNEL(variance);

#endif

TEST_SUITE(stats)
{
  TEST_HOSTDEV(mean);
  TEST_HOSTDEV(median);
  TEST_HOSTDEV(variance);
}

auto
main() -> int
{
  RUN_SUITE(stats);
  return 0;
}
