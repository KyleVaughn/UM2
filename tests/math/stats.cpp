#include <um2/math/stats.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

template <std::floating_point T>
HOSTDEV
TEST_CASE(mean)
{
  um2::Vector<T> v = {1, 2, 3, 4, 5};
  T const m = um2::mean(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(3), static_cast<T>(1e-6));
}

template <std::floating_point T>
HOSTDEV
TEST_CASE(median)
{
  um2::Vector<T> v = {1, 2, 3, 4, 5};
  T const m = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(3), static_cast<T>(1e-6));
  v.push_back(6);
  T const m2 = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m2, static_cast<T>(3.5), static_cast<T>(1e-6));
}

template <std::floating_point T>
HOSTDEV
TEST_CASE(variance)
{
  um2::Vector<T> v = {1, 2, 3, 4, 5};
  T const m = um2::variance(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(2.5), static_cast<T>(1e-6));
}

#if UM2_USE_CUDA

template <std::floating_point T>
MAKE_CUDA_KERNEL(mean, T);

template <std::floating_point T>
MAKE_CUDA_KERNEL(median, T);

template <std::floating_point T>
MAKE_CUDA_KERNEL(variance, T);

#endif

template <std::floating_point T>
TEST_SUITE(stats)
{
  TEST_HOSTDEV(mean, 1, 1, T);
  TEST_HOSTDEV(median, 1, 1, T);
  TEST_HOSTDEV(variance, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE(stats<float>);
  RUN_SUITE(stats<double>);
  return 0;
}
