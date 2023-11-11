//=============================================================================
// Summary
//=============================================================================
//
// Purpose:
// -------
// This benchmark is to test the performance of:
//  - CPU (single and multi-threaded) and GPU sorting
//  - BMI2 vs non-BMI2 instructions
//  - 32-bit vs 64-bit integers and floats
// Sorting points in morton order has a performance benefit for spatial queries, 
// but the benefit must outweigh the cost of sorting. Hence, we need to know the
// make sorting as fast as possible.
//
// Description:
// ------------
// Test the performance of sorting random points in morton order. The number of 
// points is varied from 1024 to 4 million.
//
// Results:
// --------
// Without BMI2:
// mortonSortSerial<double,uint64_t>/1024             159 us          159 us         4403
// mortonSortSerial<double,uint64_t>/4096             759 us          759 us          912
// mortonSortSerial<double,uint64_t>/32768           7643 us         7640 us           91
// mortonSortSerial<double,uint64_t>/262144         73139 us        73105 us           10
// mortonSortSerial<double,uint64_t>/2097152       680816 us       680395 us            1
// mortonSortSerial<double,uint64_t>/4194304      1423707 us      1422891 us            1
// mortonSortSerial<double,uint32_t>/1024             130 us          130 us         5386
// mortonSortSerial<double,uint32_t>/4096             619 us          619 us         1141
// mortonSortSerial<double,uint32_t>/32768           6157 us         6155 us          111
// mortonSortSerial<double,uint32_t>/262144         59393 us        59358 us           12
// mortonSortSerial<double,uint32_t>/2097152       559797 us       559312 us            1
// mortonSortSerial<double,uint32_t>/4194304      1172271 us      1171623 us            1
// mortonSortSerial<float,uint32_t>/1024              128 us          128 us         5458
// mortonSortSerial<float,uint32_t>/4096              608 us          608 us         1170
// mortonSortSerial<float,uint32_t>/32768            6041 us         6039 us          115
// mortonSortSerial<float,uint32_t>/262144          57669 us        57643 us           12
// mortonSortSerial<float,uint32_t>/2097152        541308 us       541055 us            1
// mortonSortSerial<float,uint32_t>/4194304       1153335 us      1152664 us            1
// mortonSortParallel<double,uint64_t>/1024          91.4 us         91.3 us         7708
// mortonSortParallel<double,uint64_t>/4096           169 us          168 us         4136
// mortonSortParallel<double,uint64_t>/32768          625 us          621 us         1089
// mortonSortParallel<double,uint64_t>/262144        3936 us         3932 us          175
// mortonSortParallel<double,uint64_t>/2097152      52091 us        50904 us           13
// mortonSortParallel<double,uint64_t>/4194304     119291 us       117706 us            6
// mortonSortParallel<double,uint32_t>/1024          71.4 us         71.4 us         9833
// mortonSortParallel<double,uint32_t>/4096           135 us          135 us         5145
// mortonSortParallel<double,uint32_t>/32768          502 us          500 us         1000
// mortonSortParallel<double,uint32_t>/262144        2952 us         2948 us          230
// mortonSortParallel<double,uint32_t>/2097152      47679 us        46397 us           15
// mortonSortParallel<double,uint32_t>/4194304     111632 us       109298 us            6
// mortonSortParallel<float,uint32_t>/1024           69.8 us         69.7 us         9953
// mortonSortParallel<float,uint32_t>/4096            134 us          133 us         5337
// mortonSortParallel<float,uint32_t>/32768           469 us          468 us         1484
// mortonSortParallel<float,uint32_t>/262144         2971 us         2964 us          232
// mortonSortParallel<float,uint32_t>/2097152       26233 us        26022 us           27
// mortonSortParallel<float,uint32_t>/4194304       63200 us        62039 us           11
//
// With BMI2:
// mortonSortSerial<double,uint64_t>/1024            69.8 us         69.8 us        10081
// mortonSortSerial<double,uint64_t>/4096             337 us          337 us         2073
// mortonSortSerial<double,uint64_t>/32768           3361 us         3360 us          208
// mortonSortSerial<double,uint64_t>/262144         32102 us        32089 us           22
// mortonSortSerial<double,uint64_t>/2097152       302886 us       302737 us            2
// mortonSortSerial<double,uint64_t>/4194304       628798 us       628459 us            1
// mortonSortSerial<double,uint32_t>/1024            70.2 us         70.2 us         9969
// mortonSortSerial<double,uint32_t>/4096             340 us          339 us         2073
// mortonSortSerial<double,uint32_t>/32768           3375 us         3374 us          208
// mortonSortSerial<double,uint32_t>/262144         31838 us        31820 us           22
// mortonSortSerial<double,uint32_t>/2097152       300034 us       299849 us            2
// mortonSortSerial<double,uint32_t>/4194304       636616 us       636199 us            1
// mortonSortSerial<float,uint32_t>/1024             69.5 us         69.5 us        10076
// mortonSortSerial<float,uint32_t>/4096              332 us          332 us         2108
// mortonSortSerial<float,uint32_t>/32768            3323 us         3322 us          211
// mortonSortSerial<float,uint32_t>/262144          31570 us        31559 us           22
// mortonSortSerial<float,uint32_t>/2097152        293648 us       293481 us            2
// mortonSortSerial<float,uint32_t>/4194304        621574 us       621259 us            1
// mortonSortParallel<double,uint64_t>/1024          48.1 us         47.9 us        14647
// mortonSortParallel<double,uint64_t>/4096          90.7 us         90.5 us         7759
// mortonSortParallel<double,uint64_t>/32768          325 us          323 us         2155
// mortonSortParallel<double,uint64_t>/262144        1962 us         1958 us          359
// mortonSortParallel<double,uint64_t>/2097152      43629 us        42626 us           16
// mortonSortParallel<double,uint64_t>/4194304     107143 us        99620 us            7
// mortonSortParallel<double,uint32_t>/1024          45.5 us         45.5 us        15426
// mortonSortParallel<double,uint32_t>/4096          88.7 us         88.6 us         7929
// mortonSortParallel<double,uint32_t>/32768          320 us          319 us         2205
// mortonSortParallel<double,uint32_t>/262144        1924 us         1918 us          370
// mortonSortParallel<double,uint32_t>/2097152      44358 us        43999 us           16
// mortonSortParallel<double,uint32_t>/4194304     107627 us       104609 us            7
// mortonSortParallel<float,uint32_t>/1024           44.4 us         44.3 us        15858
// mortonSortParallel<float,uint32_t>/4096           85.3 us         85.1 us         8165
// mortonSortParallel<float,uint32_t>/32768           313 us          313 us         2209
// mortonSortParallel<float,uint32_t>/262144         1826 us         1818 us          380
// mortonSortParallel<float,uint32_t>/2097152       17842 us        17713 us           40
// mortonSortParallel<float,uint32_t>/4194304       50308 us        48177 us           14
//
// With CUDA:
// mortonSortCuda<double,uint64_t>/1024            82.3 us         82.3 us         7787
// mortonSortCuda<double,uint64_t>/4096             120 us          120 us         5412
// mortonSortCuda<double,uint64_t>/32768            232 us          232 us         3317
// mortonSortCuda<double,uint64_t>/262144           817 us          811 us          868
// mortonSortCuda<double,uint64_t>/2097152         4133 us         4124 us          173
// mortonSortCuda<double,uint64_t>/4194304         8690 us         8686 us           81
// mortonSortCuda<double,uint32_t>/1024            74.2 us         74.2 us         8818
// mortonSortCuda<double,uint32_t>/4096             114 us          114 us         5876
// mortonSortCuda<double,uint32_t>/32768            221 us          221 us         3151
// mortonSortCuda<double,uint32_t>/262144           767 us          762 us          929
// mortonSortCuda<double,uint32_t>/2097152         3946 us         3940 us          179
// mortonSortCuda<double,uint32_t>/4194304         8503 us         8491 us           83
// mortonSortCuda<float,uint32_t>/1024             33.9 us         33.9 us        20983
// mortonSortCuda<float,uint32_t>/4096             58.5 us         58.5 us        11135
// mortonSortCuda<float,uint32_t>/32768             106 us          106 us         6350
// mortonSortCuda<float,uint32_t>/262144            336 us          333 us         2126
// mortonSortCuda<float,uint32_t>/2097152          1001 us          997 us          733
// mortonSortCuda<float,uint32_t>/4194304          1749 us         1745 us          402
//
// Analysis:
// ---------
// BMI2 vs non-BMI2 instructions
//    For single-threaded CPU sorting, BMI2 instructions were about 2x faster.
//    For multi-threaded CPU sorting, BMI2 instructions were faster, but not 2x.
//
// CPU (single and multi-threaded) and GPU sorting
//    Multi-threaded CPU sorting scaled approximately linearly with the number of
//    threads. Of course, for small arrays, the overhead of thread creation
//    played a significant role.
//    GPU sorting shows a very large speedup over CPU sorting, especially for
//    large arrays.
//
// 32-bit vs 64-bit integers and floats
//    The performance of 32-bit and 64-bit numbers was very similar on CPU, but
//    on GPU, 32-bit numbers were about 2x or more faster than 64-bit numbers.
//
// Conclusions:
// ------------
// 1. Since 64-bit vs 32-bit integers and floats have similar performance on CPU,
//    we should prefer 64-bit numbers for CPU sorting. This allows for a more
//    accurate spatial sorting of points. On GPU, however, 32-bit numbers are
//    much faster, so we should use 32-bit numbers for GPU sorting.
//
// 2. BMI instructions should be used for CPU sorting if available.
//    
// 3. GPU sorting is much faster than CPU sorting, so we should use GPU sorting if
//    possible.

#include "../helpers.hpp"
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/parallel/geometry/morton_sort_points.hpp>
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
  size_t const size_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>); 
  cudaError_t error = cudaSuccess;
  error = cudaMalloc(&d_points, size_in_bytes);
  CUDA_CHECK_ERROR(error);

  error = cudaMemcpy(d_points, points.data(), size_in_bytes, cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR(error);

  for (auto s : state) {
    state.PauseTiming();
    // we don't have a random_shuffle for CUDA, so just copy the points
    // to the device again
    error = cudaMemcpy(d_points, points.data(), size_in_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(error);

    state.ResumeTiming();
    um2::parallel::deviceMortonSort<U>(d_points, d_points + n);
    cudaDeviceSynchronize();
  }

  error = cudaMemcpy(after.data(), d_points, size_in_bytes, cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR(error);
  if (!std::is_sorted(after.begin(), after.end(), um2::mortonLess<U, 2, T>)) {
    std::cout << "Not sorted" << std::endl;
  }
  cudaFree(d_points);
}
#endif

BENCHMARK_TEMPLATE2(mortonSortSerial, double, uint64_t)
    ->RangeMultiplier(8)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortSerial, double, uint32_t)
  ->RangeMultiplier(8)
  ->Range(1024, npoints)
  ->Unit(benchmark::kMicrosecond);
// Lossy
//BENCHMARK_TEMPLATE2(mortonSortSerial, float, uint64_t)
//  ->RangeMultiplier(8)
//  ->Range(1024, npoints)
//  ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortSerial, float, uint32_t)
  ->RangeMultiplier(8)
  ->Range(1024, npoints)
  ->Unit(benchmark::kMicrosecond);

#if UM2_USE_TBB
BENCHMARK_TEMPLATE2(mortonSortParallel, double, uint64_t)
    ->RangeMultiplier(8)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortParallel, double, uint32_t)
    ->RangeMultiplier(8)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortParallel, float, uint32_t)    
  ->RangeMultiplier(8)    
  ->Range(1024, npoints)    
  ->Unit(benchmark::kMicrosecond);
#endif
#if UM2_USE_CUDA
BENCHMARK_TEMPLATE2(mortonSortCuda, double, uint64_t)
    ->RangeMultiplier(8)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortCuda, double, uint32_t)
    ->RangeMultiplier(8)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE2(mortonSortCuda, float, uint32_t)
    ->RangeMultiplier(8)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
