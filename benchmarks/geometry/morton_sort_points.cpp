// FINDINGS:
//  BMI2 vs non-BMI2: Use of BMI is 2x-3x faster than non-BMI
//  Single vs multi-threaded: multithreaded appears to be same for faster
//    Also, the execution policy for sort does not seem to matter
//  CUDA: Faster after 1000 points

#include <benchmark/benchmark.h>
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/common/Vector.hpp>

#include <execution>
#include <random>
#include <algorithm>
#include <parallel/algorithm>

#define D 2
#define T double 
#define U uint32_t
#define NPOINTS 1 << 18 

auto randomPoint() -> um2::Point<D, T> 
{
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(0, 1); 
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
      p[i] = dist(rng);
  }
  return p;
}

static void mortonSortSerial(benchmark::State& state) {
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  for (auto& p : points) {
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

static void mortonSortParallel(benchmark::State& state) {
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  for (auto& p : points) {
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

//static void mortonSortParallelExec(benchmark::State& state) {
//  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
//  for (auto& p : points) {
//    // cppcheck-suppress useStlAlgorithm
//    p = randomPoint();
//  }
//  // NOLINTNEXTLINE
//  for (auto s : state) {
//    std::sort(std::execution::par, points.begin(), points.end(), um2::mortonLess<U, D, T>);
//  }
//  if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, D, T>)) {
//    std::cout << "Not sorted" << std::endl;
//  }
//}

#if UM2_ENABLE_CUDA
static void mortonSortCuda(benchmark::State& state) {
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  um2::Vector<um2::Point<D, T>> after(static_cast<Size>(state.range(0)));
  for (auto& p : points) {
    // cppcheck-suppress useStlAlgorithm
    p = randomPoint();
  }
  um2::Point2<T> * d_points;    
  size_t const size_in_bytes = sizeof(um2::Point2<T>) * static_cast<size_t>(points.size());    
  cudaMalloc(&d_points, size_in_bytes);     
  cudaMemcpy(d_points, points.data(), size_in_bytes, cudaMemcpyHostToDevice);

  // NOLINTNEXTLINE
  for (auto s : state) {
    um2::deviceMortonSort<U>(d_points, d_points + points.size());
  }
  
  cudaMemcpy(after.data(), d_points, size_in_bytes, cudaMemcpyDeviceToHost);
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
