#include <um2/parallel/math/stats.hpp>

#include "../../test_macros.hpp"

#include <vector>

template <std::floating_point T>
TEST_CASE(mean)
{
  std::vector<T> const v{1, 2, 3, 4, 5};
  T const m = um2::parallel::mean(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(3), static_cast<T>(1e-6));
}

template <std::floating_point T>
TEST_CASE(variance)
{
  std::vector<T> const v{1, 2, 3, 4, 5};
  T const m = um2::parallel::variance(v.data(), v.data() + v.size());
  ASSERT_NEAR(m, static_cast<T>(2.5), static_cast<T>(1e-6));
}

template <std::floating_point T>
TEST_SUITE(stats)
{
  TEST((mean<T>));
  TEST((variance<T>));
}

auto
main() -> int
{
  RUN_SUITE(stats<float>);
  RUN_SUITE(stats<double>);
  return 0;
}
