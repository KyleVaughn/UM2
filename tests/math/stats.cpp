#include <um2/math/stats.hpp>

#include "../test_macros.hpp"

#include <vector>

template <std::floating_point T>
TEST_CASE(mean)
{
  std::vector<T> v{1, 2, 3, 4, 5};
  T const m = um2::mean(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(3), static_cast<T>(1e-6));
}

template <std::floating_point T>
TEST_CASE(median)
{
  std::vector<T> v{1, 2, 3, 4, 5};
  T const m = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(3), static_cast<T>(1e-6));
  v.push_back(6);
  T const m2 = um2::median(v.data(), v.data() + v.size());
  ASSERT_NEAR(m2, static_cast<T>(3.5), static_cast<T>(1e-6));
}

template <std::floating_point T>
TEST_CASE(variance)
{
  std::vector<T> v{1, 2, 3, 4, 5};
  T const m = um2::variance(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(2.5), static_cast<T>(1e-6));
}

template <std::floating_point T>
TEST_SUITE(stats)
{
  TEST((mean<T>));
  TEST((median<T>));
  TEST((variance<T>));
}

auto
main() -> int
{
  RUN_SUITE(stats<float>);
  RUN_SUITE(stats<double>);
  return 0;
}
