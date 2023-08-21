#pragma once

#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <vector>

#ifdef USE_GNU_PARALLEL
#  include <parallel/numeric>
#endif

namespace um2
{
template <typename T>
auto
mean(T * begin, T * end) -> T
{
  auto const count = static_cast<size_t>(std::distance(begin, end));
  assert(count != 0);

  // Using std::reduce with parallel execution policy
#ifdef USE_GNU_PARALLEL
  T total = std::reduce(std::execution::par, begin, end, T(0));
#else
  T total = std::reduce(std::execution::seq, begin, end, static_cast<T>(0));
#endif
  return total / static_cast<T>(count);
}

template <typename T>
auto
median(T * begin, T * end) -> T
{
  auto count = static_cast<size_t>(std::distance(begin, end));

  assert(count != 0);

  std::vector<T> data_copy(begin, end);

#ifdef USE_GNU_PARALLEL
  __gnu_parallel::sort(data_copy.begin(), data_copy.end());
#else
  std::sort(data_copy.begin(), data_copy.end());
#endif

  if (count % 2 == 1) { // Odd number of elements
    return data_copy[count / 2];
  } // Even number of elements
  return (data_copy[(count - 1) / 2] + data_copy[count / 2]) / 2;
}

template <typename T>
auto
variance(T * begin, T * end) -> T
{
  auto count = static_cast<size_t>(std::distance(begin, end));

  if (count < 2) {
    throw std::invalid_argument("Need at least two data points for variance.");
  }
#ifdef USE_GNU_PARALLEL
  // Calculate the mean
  T mean = std::reduce(begin, end, static_cast<T>(0)) / static_cast<double>(count);
  T sum_of_squares =
      std::reduce(std::execution::par, begin, end, static_cast<T>(0),
                  [mean](T acc, T val) { return acc + (val - mean) * (val - mean); });
#else
  T data_mean = std::reduce(std::execution::seq, begin, end, static_cast<T>(0)) /
                static_cast<double>(count);
  T sum_of_squares =
      std::reduce(begin, end, static_cast<T>(0), [data_mean](T acc, T val) {
        return acc + (val - data_mean) * (val - data_mean);
      });
#endif

  return sum_of_squares / (static_cast<double>(count));
}
} // namespace um2