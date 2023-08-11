// FINDINGS:
//  BMI2 vs non-BMI2: Use of BMI is 2x-3x faster than non-BMI
//  Single vs multi-threaded: multithreaded appears to be same or faster
//    Also, the execution policy for sort does not seem to matter
//  CUDA: Faster after 1000 points
//  double, uint64_t isn't much slower than float, uint32_t, so uint64_t is
//  preferred, since it gives better spatial resolution

#include "../helpers.hpp"
#include <um2/geometry/morton_sort_points.hpp>

#include <execution>
#include <iostream>

constexpr Size npoints = 1 << 20;
constexpr Size dim = 2;

template <typename T, typename U>
static void
mortonSortSerial(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0, 1>(n);
  std::random_device rd;
  std::mt19937 g(rd());
  // NOLINTNEXTLINE
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(points.begin(), points.end(), g);
    state.ResumeTiming();
    std::sort(points.begin(), points.end(), um2::mortonLess<U, dim, T>);
  }
  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, dim, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
}

#if _OPENMP
template <typename T, typename U>
static void
mortonSortParallel(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0, 1>(n);
  std::random_device rd;
  std::mt19937 g(rd());
  // NOLINTNEXTLINE
  for (auto s : state) {
    state.PauseTiming();
    std::shuffle(points.begin(), points.end(), g);
    state.ResumeTiming();
    __gnu_parallel::sort(points.begin(), points.end(), um2::mortonLess<U, dim, T>);
  }
  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, dim, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
}

// template <typename T, typename U>
// static void mortonSortParallelExec(benchmark::State& state)
//{
//   Size const n = static_cast<Size>(state.range(0));
//   um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0, 1>(n);
//   // NOLINTNEXTLINE
//   for (auto s : state) {
//     state.PauseTiming();
//     std::random_shuffle(points.begin(), points.end());
//     state.ResumeTiming();
//     std::sort(std::execution::par_unseq, points.begin(), points.end(),
//     um2::mortonLess<U, dim, T>);
//   }
//   if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, dim, T>)) {
//     std::cout << "Not sorted" << std::endl;
//   }
// }
#endif

#if UM2_ENABLE_CUDA
template <typename T, typename U>
static void
mortonSortCuda(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0, 1>(n);
  um2::Vector<um2::Point<dim, T>> after(n);

  um2::Point2<T> * d_points;
  transferToDevice(&d_points, points);

  // NOLINTNEXTLINE
  for (auto s : state) {
    state.PauseTiming();
    // we don't have a random_shuffle for CUDA, so just copy the points
    // to the device again
    transferToDevice(&d_points, points);
    state.ResumeTiming();
    um2::deviceMortonSort<U>(d_points, d_points + points.size());
    cudaDeviceSynchronize();
  }

  transferFromDevice(after, d_points);
  if (!std::is_sorted(after.begin(), after.end(), um2::mortonLess<U, dim, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
  cudaFree(d_points);
}
#endif

BENCHMARK_TEMPLATE2(mortonSortSerial, double, uint64_t)
    ->RangeMultiplier(4)
    ->Range(16, npoints)
    ->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE2(mortonSortSerial, float, uint32_t)
//   ->RangeMultiplier(4)
//   ->Range(16, npoints)
//   ->Unit(benchmark::kMicrosecond);

#if _OPENMP
BENCHMARK_TEMPLATE2(mortonSortParallel, double, uint64_t)
    ->RangeMultiplier(4)
    ->Range(16, npoints)
    ->Unit(benchmark::kMicrosecond);
// BENCHMARK_TEMPLATE2(mortonSortParallelExec, double, uint64_t)
//   ->RangeMultiplier(4)
//   ->Range(16, npoints)
//   ->Unit(benchmark::kMicrosecond);
#endif
#if UM2_ENABLE_CUDA
BENCHMARK_TEMPLATE2(mortonSortCuda, double, uint64_t)
    ->RangeMultiplier(4)
    ->Range(16, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
