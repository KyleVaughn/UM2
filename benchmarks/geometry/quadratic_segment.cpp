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
// closestPointOnSegmentWB<float>/262144       1230 us         1230 us          569
// closestPointOnSegmentPB<float>/262144       9093 us         9092 us           77
// bezierTriangleWB<float>/262144              86.1 us         86.1 us         8033
// bezierTrianglePB<float>/262144              1410 us         1410 us          493

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
  return q;
}

template <typename T>
HOSTDEV constexpr auto
makeSeg4() -> um2::QuadraticSegment2<T>
{
  um2::QuadraticSegment2<T> q = makeBaseSeg<T>();
  q[2][0] = static_cast<T>(2);
  q[2][1] = static_cast<T>(1);
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
BENCHMARK_TEMPLATE(closestPointOnSegmentWB, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(closestPointOnSegmentPB, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(bezierTriangleWB, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(bezierTrianglePB, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
