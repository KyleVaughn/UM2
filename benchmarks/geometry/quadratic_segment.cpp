//=============================================================================
// Summary
//=============================================================================
//
// Purpose:
// -------
// This benchmark is to test the performance of various algorithms for testing
// if a 2D point is left of a quadratic segment.
//
// Description:
// ------------
// Test the performance of the following algorithms:
// - closestPointOnSegment:
//    If the point is not in bounding box of the curve, treat the curve as a straight
//    line. Otherwise, find the point on the segment closest to the point in question.
//    Test if V0, Vclose, and P are counter-clockwise oriented.
// - (no longer used) rotatedAABB:
//    Rotate/translate the curve so that V0 and V1 are on the x-axis.
//    If the point is not in the bounding box of the curve, treat the curve as a straight
//    line. Otherwise, solve for the point with the same x-coordinate as P that is
//    closest to the curve. Test if V0, Vclose, and P are counter-clockwise oriented.
//    (In an old benchmark this was close to, but strightly slower than, bezierTriangle.)
// - bezierTriangle:
//    Rotate/translate the curve so that V0 and V1 are on the x-axis.
//    If the point is not in the triangle formed by the points of the equivalent
//    bezier triangle, treat the curve as a straight line. Otherwise, solve for the
//    point with the same x-coordinate as P that is closest to the curve. Test if
//    V0, Vclose, and P are counter-clockwise oriented.
// Test each algorithm with 32-bit and 64-bit floats.
// Test each algorithm on CPU and GPU.
// Test each algorithm for a "well behaved" curve and a "poorly behaved" curve.
//  - Well behaved: no two points share the same x-coordinate.
//  - Poorly behaved: Many points share the same x-coordinate.
//
// Diagram of a well behaved curve:
//            ....
//         ..      ..
//       ..          ..
//     ..             ..
//    ..                ..
//
// Diagram of a poorly behaved curve:
//             ....
//          ..     ..
//       ..         ..
//     ..          ..
//    ..          ..
//
//
// Test the cases using random points in a box around the curve.
//
// Results:
// --------
// closestPointOnSegmentWB<float>/262144           6893 us         6893 us          102
// closestPointOnSegmentPB<float>/262144           7146 us         7146 us           99
// bezierTriangleWB<float>/262144                  1867 us         1867 us          377
// bezierTrianglePB<float>/262144                  1824 us         1824 us          383
// closestPointOnSegmentWBCUDA<float>/262144       38.8 us         38.8 us        18103
// closestPointOnSegmentPBCUDA<float>/262144       39.1 us         39.1 us        17999
// bezierTriangleWBCUDA<float>/262144              7.89 us         7.89 us        89663
// bezierTrianglePBCUDA<float>/262144              7.89 us         7.89 us        8841
//
// closestPointOnSegmentWB<double>/262144           7139 us         7138 us           99
// closestPointOnSegmentPB<double>/262144           7174 us         7173 us           98
// bezierTriangleWB<double>/262144                  3190 us         3189 us          216
// bezierTrianglePB<double>/262144                  3191 us         3190 us          219
// closestPointOnSegmentWBCUDA<double>/262144        692 us          692 us          993
// closestPointOnSegmentPBCUDA<double>/262144        690 us          690 us         1012
// bezierTriangleWBCUDA<double>/262144               128 us          128 us         5083
// bezierTrianglePBCUDA<double>/262144               125 us          125 us         5555
//
// Analysis:
// ---------
// The bezierTriangle algorithm is faster than closestPointOnSegment on both CPU and GPU.
// floats are must faster than doubles on both CPU and GPU.
// The algorithms are about the same in speed for well behaved curves and poorly behaved curves.
//
// Conclusions:
// ------------
// Use bezierTriangle for both CPU and GPU.

#include <um2/geometry/dion.hpp>
#include "../helpers.hpp"

constexpr Size npoints = 1 << 18;
// BB of base seg is [0, 0] to [2, 1]
// BB of seg4 is [0, 0] to [2.25, 2]
constexpr int lo = 0;
constexpr int hi = 3;

template <typename T>
HOSTDEV constexpr auto
makeBaseSeg() -> um2::QuadraticSegment2<T>
{
  um2::QuadraticSegment2<T> q;
  q[0] = um2::Vec<2, T>::zero();
  q[1] = um2::Vec<2, T>::zero();
  q[2] = um2::Vec<2, T>::zero();
  q[1][0] = static_cast<T>(2);
  q[2][0] = static_cast<T>(1);
  q[2][1] = static_cast<T>(1);
  return q;
}

template <typename T>
HOSTDEV constexpr auto
makeSeg4() -> um2::QuadraticSegment2<T>
{
  um2::QuadraticSegment2<T> q = makeBaseSeg<T>();
  q[1][0] = static_cast<T>(2);
  return q;
}

template <typename T>
PURE HOSTDEV auto
isLeftClosestPointOnSegment(um2::QuadraticSegment2<T> const & q,
                            um2::Point2<T> const & p) -> bool
{
  if (!q.boundingBox().contains(p)) {
    return um2::LineSegment2<T>(q[0], q[1]).isLeft(p);
  }
  T const r = q.pointClosestTo(p);
  if (r < static_cast<T>(1e-5)) {
    return um2::LineSegment2<T>(q[0], q[1]).isLeft(p);
  }
  return um2::LineSegment2<T>(q[0], q(r)).isLeft(p);
}

template <typename T>
void
closestPointOnSegmentWB(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) {
                        return isLeftClosestPointOnSegment(seg, p);
                       });
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
void
bezierTriangleWB(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) {
                        return seg.isLeft(p);
                       });
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
void
closestPointOnSegmentPB(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeSeg4<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) {
                        return isLeftClosestPointOnSegment(seg, p);
                       });
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
void
bezierTrianglePB(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeSeg4<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) {
                        return seg.isLeft(p);
                       });
    benchmark::DoNotOptimize(i);
  }
}

#if UM2_USE_CUDA
template <typename T>
static __global__ void
closestPointOnSegmentKernel(um2::QuadraticSegment2<T> const * seg,
                            um2::Point2<T> const * points,
                            bool * bools,
                            Size const n)
{
  // Each thread is responsible for 1 point.
  Size const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  bools[index] = isLeftClosestPointOnSegment(*seg, points[index]);
}

template <typename T>
static __global__ void
bezierTriangleKernel(um2::QuadraticSegment2<T> const * seg,
                     um2::Point2<T> const * points,
                     bool * bools,
                     Size const n)
{
  // Each thread is responsible for 1 point.
  Size const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  bools[index] = seg->isLeft(points[index]);
}

template <typename T>
void
closestPointOnSegmentWBCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);

  um2::Point2<T> * points_d;
  um2::QuadraticSegment2<T> * seg_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_seg_in_bytes = sizeof(um2::QuadraticSegment2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes);
  cudaMalloc(&seg_d, size_of_seg_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(seg_d, &seg, size_of_seg_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (static_cast<uint32_t>(n) + tpb - 1) / tpb;
  for (auto s : state) {
    closestPointOnSegmentKernel<<<blocks, tpb>>>(seg_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(seg_d);
  cudaFree(bools_d);
}

template <typename T>
void
closestPointOnSegmentPBCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeSeg4<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);

  um2::Point2<T> * points_d;
  um2::QuadraticSegment2<T> * seg_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_seg_in_bytes = sizeof(um2::QuadraticSegment2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes);
  cudaMalloc(&seg_d, size_of_seg_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(seg_d, &seg, size_of_seg_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (static_cast<uint32_t>(n) + tpb - 1) / tpb;
  for (auto s : state) {
    closestPointOnSegmentKernel<<<blocks, tpb>>>(seg_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(seg_d);
  cudaFree(bools_d);
}

template <typename T>
void
bezierTriangleWBCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);

  um2::Point2<T> * points_d;
  um2::QuadraticSegment2<T> * seg_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_seg_in_bytes = sizeof(um2::QuadraticSegment2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes);
  cudaMalloc(&seg_d, size_of_seg_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(seg_d, &seg, size_of_seg_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (static_cast<uint32_t>(n) + tpb - 1) / tpb;
  for (auto s : state) {
    bezierTriangleKernel<<<blocks, tpb>>>(seg_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(seg_d);
  cudaFree(bools_d);
}

template <typename T>
void
bezierTrianglePBCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeSeg4<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);

  um2::Point2<T> * points_d;
  um2::QuadraticSegment2<T> * seg_d;
  bool * bools_d;
  size_t const size_of_points_in_bytes = static_cast<size_t>(n) * sizeof(um2::Point2<T>);
  size_t const size_of_seg_in_bytes = sizeof(um2::QuadraticSegment2<T>);
  size_t const size_of_bools_in_bytes = static_cast<size_t>(n) * sizeof(bool);
  cudaMalloc(&points_d, size_of_points_in_bytes);
  cudaMalloc(&seg_d, size_of_seg_in_bytes);
  cudaMalloc(&bools_d, size_of_bools_in_bytes);
  cudaMemcpy(points_d, points.data(), size_of_points_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(seg_d, &seg, size_of_seg_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  const uint32_t blocks = (static_cast<uint32_t>(n) + tpb - 1) / tpb;
  for (auto s : state) {
    bezierTriangleKernel<<<blocks, tpb>>>(seg_d, points_d, bools_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(points_d);
  cudaFree(seg_d);
  cudaFree(bools_d);
}

#endif

BENCHMARK_TEMPLATE(closestPointOnSegmentWB, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(closestPointOnSegmentPB, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(bezierTriangleWB, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(bezierTrianglePB, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);

#if UM2_USE_CUDA

BENCHMARK_TEMPLATE(closestPointOnSegmentWBCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(closestPointOnSegmentPBCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(bezierTriangleWBCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(bezierTrianglePBCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);

#endif

BENCHMARK_MAIN();
