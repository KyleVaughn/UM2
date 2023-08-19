// FINDINGS:
//  For points in [0, 3]^2, which tend to be in the bounding box of the segments,
//    New, well behaved (rotated aabb)   : 61.4 us
//    New, well behaved (bezier triangle): 77.5 us
//    Old, well behaved                  : 2197 us
//    Speedup range: 28x to 36x
//    New, poor behaved (rotated aabb)   : 1312 us
//    New, poor behaved (bezier triangle):  224 us
//    Old, poor behaved                  : 7027 us
//    Speedup range: 5x to 31x
//
//    The well behaved case is 26% faster using the rotated aabb, but the poorly
//    behaved case is 486% slower. Therefore, the "correct" choice depends on
//    the use case.

//  For points in [-100, 100]^2, which tend to NOT be in the bounding box of the segments,
//    New, well behaved (rotated aabb)   :  66.2 us
//    New, well behaved (bezier triangle):  76.8 us
//    Old, well behaved                  :  2704 us
//    Speedup range: 35x to 41x
//    New, poor behaved (rotated aabb)   :  921 us
//    New, poor behaved (bezier triangle):  225 us
//    Old, poor behaved                  :  2796 us
//    Speedup range: 3x to 12x
//

#include "../helpers.hpp"
#include <um2/geometry/QuadraticSegment.hpp>
#include <um2/geometry/LineSegment.hpp>

#include <iostream>
#include <thrust/complex.h>

constexpr Size npoints = 1 << 18;
// BB of base seg is [0, 0] to [2, 1]
// BB of seg4 is [0, 0] to [2.25, 2]
constexpr int lo = 0;
constexpr int hi = 3;

template <typename T>
HOSTDEV static constexpr auto
makeBaseSeg() -> um2::QuadraticSegment2<T>
{
  um2::QuadraticSegment2<T> q;
  q[0] = um2::zeroVec<2, T>();
  q[1] = um2::zeroVec<2, T>();
  q[2] = um2::zeroVec<2, T>();
  q[1][0] = static_cast<T>(2);
  return q;
}

template <typename T>
HOSTDEV static constexpr auto
makeSeg4() -> um2::QuadraticSegment2<T>
{
  um2::QuadraticSegment2<T> q = makeBaseSeg<T>();
  q[2][0] = static_cast<T>(2);
  q[2][1] = static_cast<T>(1);
  return q;
}

template <typename T>
HOSTDEV constexpr auto
isLeft(um2::QuadraticSegment2<T> const & q, um2::Point2<T> const & p) -> bool
{
  return q.isLeft(p);
}

// Note this is a stand-in for the old version of the routine. It was quite unreadable,
// but effectively the same as below. Surprisingly, this is faster than the old version.
template <typename T>
PURE HOSTDEV auto
isLeftOld(um2::QuadraticSegment2<T> const & q, um2::Point2<T> const & p) -> bool
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
static void
isLeftBenchWellBehaved(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  //  auto const seg = makeSeg4();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) { return isLeft(seg, p); });
  }
  if (i == 0) {
    std::cout << i << std::endl;
  }
}

template <typename T>
static void
isLeftBenchPoorlyBehaved(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeSeg4<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) { return isLeft(seg, p); });
  }
  if (i == 0) {
    std::cout << i << std::endl;
  }
}

template <typename T>
static void
isLeftOldWellBehavedBench(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) { return isLeftOld(seg, p); });
  }
  if (i == 0) {
    std::cout << i << std::endl;
  }
}

template <typename T>
static void
isLeftOldPoorlyBehavedBench(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeSeg4<T>();
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  int64_t i = 0;
  for (auto s : state) {
    i += std::count_if(points.begin(), points.end(),
                       [&seg](auto const & p) { return isLeftOld(seg, p); });
  }
  if (i == 0) {
    std::cout << i << std::endl;
  }
}

BENCHMARK_TEMPLATE(isLeftBenchWellBehaved, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(isLeftBenchPoorlyBehaved, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(isLeftOldWellBehavedBench, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(isLeftOldPoorlyBehavedBench, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
