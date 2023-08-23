#include <um2/math/Stats.hpp>
#include <um2/stdlib/Vector.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(mean_calculation)
{
  um2::Vector<T> v{1, 2, 3};
  ASSERT_NEAR(um2::mean(v.begin(), v.end()), 2, static_cast<T>(1e-6));
}

template <typename T>
TEST_CASE(median_calculation)
{
  um2::Vector<T> v{1, 2, 3};
  ASSERT_NEAR(um2::median(v.begin(), v.end()), 2, static_cast<T>(1e-6));
}

template <typename T>
TEST_CASE(variance_calculation)
{
  um2::Vector<T> v{1, 2, 3};
  ASSERT_NEAR(um2::variance(v.begin(), v.end()), 2, static_cast<T>(1e-6));
}

template <typename T>
TEST_CASE(histogram)
{
  std::vector<T> const v{1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10};
  um2::simpleUnicodeHistogram(v);
}

template <typename T>
TEST_SUITE(stats)
{
  TEST(mean_calculation<T>)
  TEST(median_calculation<T>)
  TEST(histogram<T>)
}

auto
main() -> int
{
  RUN_SUITE((stats<float>));
  RUN_SUITE((stats<double>));
  return 0;
}