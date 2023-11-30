//=============================================================================
// Summary
//=============================================================================
//
// Purpose:
// -------
// This benchmark is to test the performance of various algorithms for testing
// if a 2D point is inside a 2D triangle.
//
// Description:
// ------------
// Test the performance of:
// - Computing the barycentric coordinates of a point
// - Checking if the points of an edge and the point are counter-clockwise oriented
// - Same as above, but without logical short-circuiting
// using random points in a box around the triangle. Depending on the size of
// the box, the percentage of points inside the triangle will vary, which may
// effect branch prediction.
//
// Results:
// --------
// Omitting all but the largest array size:
// For points in the box [-10, 10] x [-10, 10]:
// containsBary<double>/4194304                 4858 us         4858 us          137
// containsCCW<double>/4194304                 25221 us        25220 us           28
// containsCCWNoShort<double>/4194304           5398 us         5398 us          130
// 
// containsBary<float>/4194304                  2185 us         2185 us          316
// containsCCW<float>/4194304                  21576 us        21576 us           32
// containsCCWNoShort<float>/4194304            2340 us         2340 us          274
// 
// containsBaryCUDA<double>/4194304              384 us          384 us         1827
// containsCCWCUDA<double>/4194304               360 us          360 us         1948
// containsCCWNoShortCUDA<double>/4194304        397 us          397 us         1768
// 
// containsBaryCUDA<float>/4194304              58.7 us         58.7 us        11930
// containsCCWCUDA<float>/4194304               54.9 us         54.9 us        13032
// containsCCWNoShortCUDA<float>/4194304        55.5 us         55.5 us        12200
//
// For points in the box [-1, 1] x [-1, 1]:
// containsBary<double>/4194304                 4968 us         4968 us          137
// containsCCW<double>/4194304                 24455 us        24452 us           29
// containsCCWNoShort<double>/4194304           5471 us         5471 us          124
//
// containsBary<float>/4194304                  2234 us         2234 us          314
// containsCCW<float>/4194304                  20365 us        20364 us           34
// containsCCWNoShort<float>/4194304            2288 us         2288 us          289
//
// containsBaryCUDA<double>/4194304              391 us          391 us         1823
// containsCCWCUDA<double>/4194304               358 us          358 us         1952
// containsCCWNoShortCUDA<double>/4194304        358 us          358 us         1957
//
// containsBaryCUDA<float>/4194304              60.2 us         60.2 us        11819
// containsCCWCUDA<float>/4194304               54.5 us         54.5 us        12991
// containsCCWNoShortCUDA<float>/4194304        54.5 us         54.5 us        12958
//
// Analysis:
// ---------
// CPU:
//   containsBary is the fastest algorithm for both 32-bit and 64-bit numbers.
//
// GPU:
//   containsCCWCUDA and the version without logical short-circuiting are faster
//   than containsBaryCUDA, but not by much.
//
// 32-bit vs 64-bit:
//   32-bit floats are faster than 64-bit floats on CPU by around a factor of 2.
//   On GPU, 32-bit floats are faster than 64-bit floats by around a factor of 7.
//
// Conclusions:
// ------------
// 1. 32-bit numbers are faster than 64-bit numbers on both CPU and GPU, so
//    we should prefer 32-bit numbers.
//
// 2. containsBary is the fastest algorithm on CPU, and is only slightly slower
//    on GPU, so we should use containsBary.

#include "../helpers.hpp"
#include <um2/geometry/polygon.hpp>

constexpr Size npoints = 1 << 22;
constexpr int lo = -1;
constexpr int hi = 1;

template <typename T>
HOSTDEV constexpr auto
bary(um2::Triangle2<T> const & tri, um2::Point2<T> const & p) -> bool
{
  um2::Vec2<T> const a = tri[1] - tri[0];
  um2::Vec2<T> const b = tri[2] - tri[0];
  um2::Vec2<T> const c = p - tri[0];
  T const invdet_ab = 1 / a.cross(b);
  T const r = c.cross(b) * invdet_ab;
  T const s = a.cross(c) * invdet_ab;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
}

template <typename T>
HOSTDEV constexpr auto
ccw(um2::Triangle2<T> const & tri, um2::Point2<T> const & p) -> bool
{
  return um2::areCCW(tri[0], tri[1], p) && um2::areCCW(tri[1], tri[2], p) &&
         um2::areCCW(tri[2], tri[0], p);
}

template <typename T>
HOSTDEV constexpr auto
ccwNoShort(um2::Triangle2<T> const & tri, um2::Point2<T> const & p) -> bool
{
  bool const b0 = um2::areCCW(tri[0], tri[1], p);
  bool const b1 = um2::areCCW(tri[1], tri[2], p);
  bool const b2 = um2::areCCW(tri[2], tri[0], p);
  return b0 && b1 && b2;
}

template <typename T>
void
containsBary(benchmark::State & state)
{
  constexpr um2::Point2<T> p0{0, 0};
  constexpr um2::Point2<T> p1{1, 0};
  constexpr um2::Point2<T> p2{0, 1};
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Triangle2<T> const tri(p0, p1, p2);
  int64_t i = 0;
  for (auto s : state) {
    for (auto const & p : points) {
      if (bary(tri, p)) {
        ++i;
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
void
containsCCW(benchmark::State & state)
{
  constexpr um2::Point2<T> p0{0, 0};
  constexpr um2::Point2<T> p1{1, 0};
  constexpr um2::Point2<T> p2{0, 1};
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Triangle2<T> const tri(p0, p1, p2);
  int64_t i = 0;
  for (auto s : state) {
    for (auto const & p : points) {
      if (ccw(tri, p)) {
        ++i;
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
void
containsCCWNoShort(benchmark::State & state)
{
  constexpr um2::Point2<T> p0{0, 0};
  constexpr um2::Point2<T> p1{1, 0};
  constexpr um2::Point2<T> p2{0, 1};
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Triangle2<T> const tri(p0, p1, p2);
  int64_t i = 0;
  for (auto s : state) {
    for (auto const & p : points) {
      if (ccwNoShort(tri, p)) {
        ++i;
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

#if UM2_USE_CUDA
template <typename T>
static __global__ void
baryKernel(um2::Triangle2<T> * tri, um2::Point2<T> * points, bool * bools, Size const n)
{
  // Each thread is responsible for 1 point.
  Size const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  bools[index] = bary(*tri, points[index]);
}

template <typename T>
static __global__ void
ccwKernel(um2::Triangle2<T> * tri, um2::Point2<T> * points, bool * bools, Size const n)
{
  // Each thread is responsible for 1 point.
  Size const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  bools[index] = ccw(*tri, points[index]);
}

template <typename T>
static __global__ void
ccwNoShortKernel(um2::Triangle2<T> * tri, um2::Point2<T> * points, bool * bools, Size const n)
{
  // Each thread is responsible for 1 point.
  Size const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  bools[index] = ccwNoShort(*tri, points[index]);
}

template <typename T>    
void    
containsBaryCUDA(benchmark::State & state)    
{    
  constexpr um2::Point2<T> p0{0, 0};
  constexpr um2::Point2<T> p1{1, 0};
  constexpr um2::Point2<T> p2{0, 1};
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Triangle2<T> const tri(p0, p1, p2);

  um2::Point2<T> * points_d;
  um2::Triangle2<T> * tri_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_tri_in_bytes = sizeof(um2::Triangle2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes); 
  cudaMalloc(&tri_d, size_of_tri_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(tri_d, &tri, size_of_tri_in_bytes, cudaMemcpyHostToDevice);
  //cudaMemcpy(bools_d, bools.data(), size_of_bools_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (n + tpb - 1) / tpb;
  for (auto s : state) {
    baryKernel<<<blocks, tpb>>>(tri_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(tri_d);
  cudaFree(bools_d);
}

template <typename T>    
void    
containsCCWCUDA(benchmark::State & state)    
{    
  constexpr um2::Point2<T> p0{0, 0};
  constexpr um2::Point2<T> p1{1, 0};
  constexpr um2::Point2<T> p2{0, 1};
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Triangle2<T> const tri(p0, p1, p2);

  um2::Point2<T> * points_d;
  um2::Triangle2<T> * tri_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_tri_in_bytes = sizeof(um2::Triangle2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes); 
  cudaMalloc(&tri_d, size_of_tri_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(tri_d, &tri, size_of_tri_in_bytes, cudaMemcpyHostToDevice);
  //cudaMemcpy(bools_d, bools.data(), size_of_bools_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (n + tpb - 1) / tpb;
  for (auto s : state) {
    ccwKernel<<<blocks, tpb>>>(tri_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(tri_d);
  cudaFree(bools_d);
}

template <typename T>    
void    
containsCCWNoShortCUDA(benchmark::State & state)    
{    
  constexpr um2::Point2<T> p0{0, 0};
  constexpr um2::Point2<T> p1{1, 0};
  constexpr um2::Point2<T> p2{0, 1};
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Triangle2<T> const tri(p0, p1, p2);

  um2::Point2<T> * points_d;
  um2::Triangle2<T> * tri_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_tri_in_bytes = sizeof(um2::Triangle2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes); 
  cudaMalloc(&tri_d, size_of_tri_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(tri_d, &tri, size_of_tri_in_bytes, cudaMemcpyHostToDevice);
  //cudaMemcpy(bools_d, bools.data(), size_of_bools_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (n + tpb - 1) / tpb;
  for (auto s : state) {
    ccwNoShortKernel<<<blocks, tpb>>>(tri_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(tri_d);
  cudaFree(bools_d);
}
#endif

BENCHMARK_TEMPLATE(containsBary, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCW, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCWNoShort, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(containsBary, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCW, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCWNoShort, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);

#if UM2_USE_CUDA
BENCHMARK_TEMPLATE(containsBaryCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCWCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCWNoShortCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(containsBaryCUDA, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCWCUDA, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCWNoShortCUDA, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
