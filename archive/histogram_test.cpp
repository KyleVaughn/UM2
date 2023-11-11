#include <um2/common/histogram.hpp>
#include <um2/stdlib/Vector.hpp>

#include "../test_macros.hpp"

#include <random>

template <typename T>
TEST_CASE(vector_constructor)
{
  // Generate random data
  size_t const n = 100;
  um2::Vector<T> data(n);
  // NOLINTNEXTLINE
  std::default_random_engine generator;
  std::normal_distribution<T> distribution(0.0, 1.0);
  std::generate(data.begin(), data.end(), [&]() { return distribution(generator); });
  std::sort(data.begin(), data.end());
  um2::printHistogram(data.begin(), data.end(), 11);
}

template <typename T>
TEST_SUITE(histogram)
{
  TEST((vector_constructor<T>));
}

auto
main() -> int
{
  RUN_SUITE((histogram<float>));
  RUN_SUITE((histogram<double>));
  return 0;
}