// FINDINGS:
//  BMI2 vs non-BMI2: Use of BMI is 2x-3x faster than non-BMI
//  Single vs multi-threaded: multithreaded appears to be same or faster
//    Also, the execution policy for sort does not seem to matter
//  CUDA: Faster after 1000 points

#include "../helpers.hpp"
#include <um2/geometry/morton_sort_points.hpp>

#include <execution>
#include <iostream>
#include <random>

// NOLINTBEGIN
#define D       2
#define T       double
#define U       uint32_t
#define NPOINTS 1 << 18
// NOLINTEND

auto
randomPoint() -> um2::Point<D, T>
{
  // NOLINTNEXTLINE
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(0, 1);
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = dist(rng);
  }
  return p;
}

static void
mortonSortSerial(benchmark::State & state)
{
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  for (auto & p : points) {
    // cppcheck-suppress useStlAlgorithm
    p = randomPoint();
  }
  // NOLINTNEXTLINE
  for (auto s : state) {
    std::sort(points.begin(), points.end(), um2::mortonLess<U, D, T>);
  }
  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, D, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
}

static void
mortonSortParallel(benchmark::State & state)
{
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  for (auto & p : points) {
    // cppcheck-suppress useStlAlgorithm
    p = randomPoint();
  }
  // NOLINTNEXTLINE
  for (auto s : state) {
    __gnu_parallel::sort(points.begin(), points.end(), um2::mortonLess<U, D, T>);
  }
  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, D, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
}

// static void mortonSortParallelExec(benchmark::State& state) {
//   um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
//   for (auto& p : points) {
//     // cppcheck-suppress useStlAlgorithm
//     p = randomPoint();
//   }
//   // NOLINTNEXTLINE
//   for (auto s : state) {
//     std::sort(std::execution::par, points.begin(), points.end(), um2::mortonLess<U, D,
//     T>);
//   }
//   if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, D, T>)) {
//     std::cout << "Not sorted" << std::endl;
//   }
// }

#if UM2_ENABLE_CUDA
static void
mortonSortCuda(benchmark::State & state)
{
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  um2::Vector<um2::Point<D, T>> after(static_cast<Size>(state.range(0)));
  for (auto & p : points) {
    p = randomPoint();
  }
  um2::Point2<T> * d_points;
  transferToDevice(&d_points, points);

  // NOLINTNEXTLINE
  for (auto s : state) {
    um2::deviceMortonSort<U>(d_points, d_points + points.size());
  }

  transferFromDevice(after, d_points);
  if (!std::is_sorted(after.begin(), after.end(), um2::mortonLess<U, D, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
  cudaFree(d_points);
}
#endif

BENCHMARK(mortonSortSerial)->RangeMultiplier(4)->Range(16, NPOINTS);
BENCHMARK(mortonSortParallel)->RangeMultiplier(4)->Range(16, NPOINTS);
// BENCHMARK(mortonSortParallelExec)->RangeMultiplier(4)->Range(16, NPOINTS);
#if UM2_ENABLE_CUDA
BENCHMARK(mortonSortCuda)->RangeMultiplier(4)->Range(16, NPOINTS);
#endif
BENCHMARK_MAIN();
