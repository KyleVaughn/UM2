//=============================================================================
// Findings
//=============================================================================

#include "../helpers.hpp"
#include <um2/stdlib/algorithm.hpp>

#include <iostream>

constexpr Size nvals = 1 << 12;

template <typename T>
void
insertionSort(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> vals =
      makeVectorOfRandomFloats(n, static_cast<T>(0), static_cast<T>(100));
  std::random_device rd;
  std::mt19937 g(rd());
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(vals.begin(), vals.end(), g);
    state.ResumeTiming();
    um2::insertionSort(vals.begin(), vals.end());
  }
  if (!std::is_sorted(vals.begin(), vals.end())) {
    std::cout << "Not sorted" << std::endl;
  }
}

template <typename T>
void
stdSort(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> vals =
      makeVectorOfRandomFloats(n, static_cast<T>(0), static_cast<T>(100));
  std::random_device rd;
  std::mt19937 g(rd());
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(vals.begin(), vals.end(), g);
    state.ResumeTiming();
    std::sort(vals.begin(), vals.end());
  }
  if (!std::is_sorted(vals.begin(), vals.end())) {
    std::cout << "Not sorted" << std::endl;
  }
}

#if UM2_USE_TBB
template <typename T>
void
stdSortPar(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> vals =
      makeVectorOfRandomFloats(n, static_cast<T>(0), static_cast<T>(100));
  std::random_device rd;
  std::mt19937 g(rd());
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(vals.begin(), vals.end(), g);
    state.ResumeTiming();
    std::sort(std::execution::par_unseq, vals.begin(), vals.end());
  }
  if (!std::is_sorted(vals.begin(), vals.end())) {
    std::cout << "Not sorted" << std::endl;
  }
}
#endif

BENCHMARK_TEMPLATE1(insertionSort, float)->RangeMultiplier(2)->Range(16, nvals);
BENCHMARK_TEMPLATE1(stdSort, float)->RangeMultiplier(2)->Range(16, nvals);
#if UM2_USE_TBB
BENCHMARK_TEMPLATE1(stdSortPar, float)->RangeMultiplier(2)->Range(16, nvals);
#endif
BENCHMARK_MAIN();
