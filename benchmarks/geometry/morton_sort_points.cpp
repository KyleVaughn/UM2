//=============================================================================
// Findings
//=============================================================================
// BMI2 vs non-BMI2:
//  On i7-12800H, Use of BMI is 1.5-3x faster than non-BMI.
//  Approximately 165 ns vs 97 ns per point
// double, uint64_t vs float, uint32_t:
//  double, uint64_t takes 97 ns per point
//  float, uint32_t takes 88 ns per point
//  This is a 10% improvement
// Single vs multi-threaded:
//  multithreaded appears to be same or faster than single-threaded
//  There is high computational intensity, so this is not surprising
// CUDA:
//  For 4 million points:
//  double, uint64_t single threaded = 589740 us = 140 ns per point
//  double, uint64_t multi threaded = 60345 us = 14 ns per point
//    (nearly perfect scaling on 10 core)
//  double, uint64_t = 111 us = 0.026 ns per point
//  float, uint32_t = 108 us = 0.026 ns per point
//  CUDA is 5,000x faster than single threaded CPU

#include "../helpers.hpp"
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/parallel/geometry/morton_sort_points.hpp>

#include <execution>
#include <iostream>

constexpr Size npoints = 1 << 22;

template <typename T, typename U>
void
mortonSortSerial(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({0, 0}, {1, 1});
  um2::Vector<um2::Point2<T>> points = makeVectorOfRandomPoints(n, box);
  std::random_device rd;
  std::mt19937 g(rd());
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(points.begin(), points.end(), g);
    state.ResumeTiming();
    mortonSort<U>(points.begin(), points.end());
  }
  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, 2, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
}

#if UM2_USE_TBB
template <typename T, typename U>
void
mortonSortParallel(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({0, 0}, {1, 1});
  um2::Vector<um2::Point2<T>> points = makeVectorOfRandomPoints(n, box);
  std::random_device rd;
  std::mt19937 g(rd());
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(points.begin(), points.end(), g);
    state.ResumeTiming();
    um2::parallel::mortonSort<U>(points.begin(), points.end());
  }
  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, 2, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
}
#endif

#if UM2_USE_CUDA
template <typename T, typename U>
void
mortonSortCuda(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({0, 0}, {1, 1});
  um2::Vector<um2::Point2<T>> points = makeVectorOfRandomPoints(n, box);
  um2::Vector<um2::Point2<T>> after(n);

  um2::Point2<T> * d_points;
  transferToDevice(&d_points, points);

  for (auto s : state) {
    state.PauseTiming();
    // we don't have a random_shuffle for CUDA, so just copy the points
    // to the device again
    transferToDevice(&d_points, points);
    state.ResumeTiming();
    um2::parallel::deviceMortonSort<U>(d_points, d_points + points.size());
    cudaDeviceSynchronize();
  }

  transferFromDevice(after, d_points);
  if (!std::is_sorted(after.begin(), after.end(), um2::mortonLess<U, 2, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
  cudaFree(d_points);
}
#endif

BENCHMARK_TEMPLATE2(mortonSortSerial, double, uint64_t)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE2(mortonSortSerial, float, uint32_t)
//   ->RangeMultiplier(4)
//   ->Range(1024, npoints)
//   ->Unit(benchmark::kMicrosecond);
//
#if UM2_USE_TBB
BENCHMARK_TEMPLATE2(mortonSortParallel, double, uint64_t)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
#if UM2_USE_CUDA
BENCHMARK_TEMPLATE2(mortonSortCuda, float, uint32_t)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortCuda, double, uint64_t)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
