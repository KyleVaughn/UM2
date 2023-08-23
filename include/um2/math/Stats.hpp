#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <execution>
#include <iomanip>
#include <iostream>
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

template <std::floating_point T>
void
simpleUnicodeHistogram(const std::vector<T> & x, int nbins = -1, int plot_width = 30,
                       bool show_counts = true, double outlier_quantile = 0.999,
                       const std::string & xlabel = "", const std::string & ylabel = "")
{
  const std::vector<std::string> blocks = {" ", "▏", "▎", "▍", "▌",
                                           "▋", "▊", "▉", "█", "█"};
  if (nbins == -1) {
    nbins = static_cast<int>(ceil(log2(static_cast<double>(x.size())) + 1));
  }

  T l = *std::min_element(x.begin(), x.end());
  T M = *std::max_element(x.begin(), x.end());
  // Note: std::nextafter works for float and double, but not necessarily for all T
  // It's retained here but might need a special handling for non-floating-point T types.
  l = std::nextafter(l, l - 1);

  std::vector<T> sorted_x = x;
  std::sort(sorted_x.begin(), sorted_x.end());
  T Q = sorted_x[static_cast<int>(sorted_x.size() * outlier_quantile)];

  double initial_dx = (M - l) / nbins;
  bool truncate = M - Q > 2 * initial_dx;
  T u = truncate ? Q : M;

  std::vector<int> hist_counts(static_cast<uint64_t>(nbins), 0);
  double dx = truncate ? (u - l) / (nbins - 1) : initial_dx;

  for (T xi : x) {
    int index = static_cast<int>(ceil((xi - l) / dx));
    if (1 <= index && index <= nbins) {
      hist_counts[static_cast<uint64_t>(index - 1)]++;
    } else {
      hist_counts[static_cast<uint64_t>(nbins - 1)]++;
    }
  }

  std::vector<T> bin_edges;
  if (truncate) {
    for (int i = 0; i < nbins; ++i) {
      bin_edges.push_back(l + i * (u - l) / (nbins - 1));
    }
    bin_edges.push_back(M);
  } else {
    for (int i = 0; i <= nbins; ++i) {
      bin_edges.push_back(l + i * (u - l) / nbins);
    }
  }

  int d = static_cast<int>(ceil(-log10(u - l))) + 1;
  double const scale = static_cast<double>(plot_width) /
                       (*std::max_element(hist_counts.begin(), hist_counts.end()));

  if (!ylabel.empty()) {
    std::cout << ylabel << "\n";
  }

  for (int i = 0; i < nbins; ++i) {
    double const nblocks = hist_counts[static_cast<uint64_t>(i)] * scale;
    std::string black_string;
    for (int j = 0; j < static_cast<int>(floor(nblocks)); ++j) {
      black_string += "█";
    }
    std::string const block_string =
        black_string + blocks[static_cast<uint64_t>(
                           static_cast<int>(ceil((nblocks - floor(nblocks)) * 8)))];

    std::cout << " (" << std::setprecision(d + static_cast<int>(ceil(log10(nbins))) - 1)
              << bin_edges[i] << " - " << bin_edges[i + 1] << ")  " << block_string;

    if (show_counts) {
      std::cout << " " << hist_counts[static_cast<uint64_t>(i)];
    }

    std::cout << '\n';
  }

  if (!xlabel.empty()) {
    std::cout << "\n"
              << std::string(
                     static_cast<uint64_t>(std::max(
                         plot_width / 2 + 6 - static_cast<int>(xlabel.length()) / 2, 0)),
                     ' ')
              << xlabel << '\n';
  }
}

} // namespace um2