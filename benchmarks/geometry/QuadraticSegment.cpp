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

#include <iostream>
#include <thrust/complex.h>

constexpr Size dim = 2;
constexpr Size npoints = 1 << 18;
// BB of base seg is [0, 0] to [2, 1]
// BB of seg4 is [0, 0] to [2.25, 2]
constexpr int lo = -100;
constexpr int hi = 100;

// NOLINTBEGIN(readability-*)

template <typename T>
HOSTDEV static constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<dim, T>
{
  um2::QuadraticSegment<dim, T> q;
  q[0] = um2::zeroVec<dim, T>();
  q[1] = um2::zeroVec<dim, T>();
  q[2] = um2::zeroVec<dim, T>();
  q[1][0] = static_cast<T>(2);
  return q;
}

template <typename T>
HOSTDEV static constexpr auto
makeSeg4() -> um2::QuadraticSegment<dim, T>
{
  um2::QuadraticSegment<dim, T> q = makeBaseSeg<T>();
  q[2][0] = static_cast<T>(2);
  q[2][1] = static_cast<T>(1);
  return q;
}

template <typename T>
HOSTDEV constexpr auto
isLeft(um2::QuadraticSegment<dim, T> const & seg, um2::Point<dim, T> const & p) -> bool
{
  return seg.isLeft(p);
}

template <typename T>
PURE HOSTDEV auto
isLeftOld(um2::QuadraticSegment<dim, T> const & Q, um2::Point<dim, T> const & p) -> bool
{
  um2::Vec2<T> const v13 = Q[2] - Q[0];
  um2::Vec2<T> const v23 = Q[2] - Q[1];
  um2::Vec2<T> const va = -2 * (v13 + v23);
  um2::Vec2<T> const vb = 3 * v13 + v23;
  um2::Vec2<T> const vr(-vb[0] / (2 * va[0]), -vb[1] / (2 * va[1]));
  um2::Vec2<T> vmin = min(Q[0], Q[1]);
  um2::Vec2<T> vmax = max(Q[0], Q[1]);
  if (0 <= vr[0] && vr[0] <= 1) {
    T const x_stationary = vr[0] * vr[0] * va[0] + vr[0] * vb[0] + Q[0][0];
    vmin[0] = um2::min(vmin[0], x_stationary);
    vmax[0] = um2::max(vmax[0], x_stationary);
  }
  if (0 <= vr[1] && vr[1] <= 1) {
    T const y_stationary = vr[1] * vr[1] * va[1] + vr[1] * vb[1] + Q[0][1];
    vmin[1] = um2::min(vmin[1], y_stationary);
    vmax[1] = um2::max(vmax[1], y_stationary);
  }
  T const a = 2 * va.squaredNorm();

  // If the point is outside of the box, or |a| is small, then we can
  // treat the segment as a line segment.
  bool const in_box =
      vmin[0] <= p[0] && p[0] <= vmax[0] && vmin[1] <= p[1] && p[1] <= vmax[1];
  if (!in_box || a < static_cast<T>(1e-7)) {
    return 0 <= (Q[1] - Q[0]).cross(p - Q[0]);
  }

  T const b = 3 * (va.dot(vb));

  um2::Vec2<T> const vw = p - Q[0];

  T const d = -(vb.dot(vw));
  //  if (d < static_cast<T>(1e-7)) { // one root is 0
  //      return 0 <= cross(Q[1] - Q[0], p - Q[0]);
  //  }

  T const c = vb.squaredNorm() - 2 * (va.dot(vw));

  // Lagrange's method
  T const s0 = -b / a;
  T const e1 = s0;
  T const e2 = c / a;
  T const e3 = -d / a;
  T const A = 2 * e1 * e1 * e1 - 9 * e1 * e2 + 27 * e3;
  T const B = e1 * e1 - 3 * e2;
  T const disc = A * A - 4 * B * B * B;
  if (0 < disc) { // One real root
    const T s1 = cbrt((A + sqrt(disc)) / 2);
    T const s2 = abs(s1) < static_cast<T>(1e-7) ? 0 : B / s1;
    T const r = (s0 + s1 + s2) / 3;
    if (0 <= r && r <= 1) {
      return 0 <= Q.jacobian(r).cross(p - Q(r));
    }
    return 0 <= (Q[1] - Q[0]).cross(p - Q[0]);
  }
  thrust::complex<T> const t1 =
      exp(static_cast<T>(0.3333333333333333) *
          log(static_cast<T>(0.5) * (A + sqrt(static_cast<thrust::complex<T>>(disc)))));
  thrust::complex<T> const t2 = t1 == static_cast<thrust::complex<T>>(0)
                                    ? static_cast<thrust::complex<T>>(0)
                                    : B / t1;
  // The constructor for thrust::complex<T> is constexpr, but sqrt is not.
  // zeta1 = (-1/2, sqrt(3)/2)
  const thrust::complex<T> zeta1(static_cast<T>(-0.5),
                                 static_cast<T>(0.8660254037844386));
  const thrust::complex<T> zeta2(conj(zeta1));

  // Pick the point closest to p
  T r = 0;
  T dist = static_cast<T>(1e10);

  T const r1 = ((s0 + t1 + t2) / 3).real();
  if (0 <= r1 && r1 <= 1) {
    T const d1 = p.squaredDistanceTo(Q(r1));
    if (d1 < dist) {
      dist = d1;
      r = r1;
    }
  }

  T const r2 = ((s0 + zeta2 * t1 + zeta1 * t2) / 3).real();
  if (0 <= r2 && r2 <= 1) {
    T const d2 = p.squaredDistanceTo(Q(r2));
    if (d2 < dist) {
      dist = d2;
      r = r2;
    }
  }

  T const r3 = ((s0 + zeta1 * t1 + zeta2 * t2) / 3).real();
  if (0 <= r3 && r3 <= 1) {
    T const d3 = p.squaredDistanceTo(Q(r3));
    if (d3 < dist) {
      r = r3;
    }
  }
  return 0 <= Q.jacobian(r).cross(p - Q(r));
}

// NOLINTEND(readability-*)

template <typename T>
static void
isLeftBenchWellBehaved(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const seg = makeBaseSeg<T>();
  //  auto const seg = makeSeg4();
  auto const points = makeVectorOfRandomPoints<2, T, lo, hi>(n);
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
  auto const points = makeVectorOfRandomPoints<2, T, lo, hi>(n);
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
  auto const points = makeVectorOfRandomPoints<2, T, lo, hi>(n);
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
  auto const points = makeVectorOfRandomPoints<2, T, lo, hi>(n);
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
