#include <um2/math/stats.hpp>

#include <um2/common/cast_if_not.hpp>
#include <um2/config.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

#include <algorithm>
#include <cstdint>
#include <random>

Float constexpr eps = castIfNot<Float>(1e-6);

TEST_CASE(sum)
{
  Int const n = 16384;
  Int const soln = (n / 2) * (n + 1);
  um2::Vector<Float> v(n);
  for (Int i = 0; i < n; ++i) {
    v[i] = static_cast<Float>(i + 1);
  }
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 g(seed);
  std::shuffle(v.begin(), v.end(), g);
  Float const vsum = um2::sum(v.cbegin(), v.cend());
  // Assert that no meaningful rounding error has occurred
  // Note this is not the case for a naive implementation or
  // for fast math optimizations, which may reorder operations
#if !UM2_ENABLE_FASTMATH
  ASSERT(um2::abs(vsum - static_cast<Float>(soln)) < 1);
#else
  ASSERT(um2::abs(vsum - static_cast<Float>(soln)) < 100);
#endif
}

HOSTDEV
TEST_CASE(mean)
{
  um2::Vector<Float> v = {1, 2, 3, 4, 5};
  Float const m = um2::mean(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, castIfNot<Float>(3), eps);
}

HOSTDEV
TEST_CASE(median)
{
  um2::Vector<Float> v = {1, 2, 3, 4, 5};
  Float const m = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, castIfNot<Float>(3), eps);
  v.push_back(6);
  Float const m2 = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m2, castIfNot<Float>(3.5), eps);
}

HOSTDEV
TEST_CASE(variance)
{
  um2::Vector<Float> v = {1, 2, 3, 4, 5};
  Float const m = um2::variance(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, castIfNot<Float>(2.5), eps);
}

#if UM2_USE_CUDA

MAKE_CUDA_KERNEL(mean);
MAKE_CUDA_KERNEL(median);
MAKE_CUDA_KERNEL(variance);

#endif

TEST_SUITE(stats)
{
  TEST(sum);
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
