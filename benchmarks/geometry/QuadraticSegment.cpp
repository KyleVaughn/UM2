// FINDINGS: The new algorithm is 3x to 30x faster depending on the shape of the segment.
//           The best case (common) is 30x faster. The worst case (rare) is 3x faster.

#include <benchmark/benchmark.h>
#include <um2/common/Vector.hpp>
#include <um2/geometry/QuadraticSegment.hpp>

#include <thrust/complex.h>
#include <random>

#define D 2
#define T float
#define NPOINTS 1 << 16

auto randomPoint() -> um2::Point<D, T>
{
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(-10, 10);
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
      p[i] = dist(rng);
  }
  return p;
}

HOSTDEV static constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q;
  q[0] = um2::zeroVec<D, T>();
  q[1] = um2::zeroVec<D, T>();
  q[2] = um2::zeroVec<D, T>();
  q[1][0] = static_cast<T>(2);
  return q;
}

HOSTDEV static constexpr auto
makeSeg4() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg();
  q[2][0] = static_cast<T>(2);
  q[2][1] = static_cast<T>(1);
  return q;
}

auto makePoints(Size n) -> um2::Vector<um2::Point<D, T>>
{
  um2::Vector<um2::Point<D, T>> points(n);
  for (auto & p : points) {
    p = randomPoint();
  }
  return points;
}

HOSTDEV constexpr auto
isLeft(um2::QuadraticSegment<D, T> const & seg, um2::Point<D, T> const & p) -> bool
{
  return seg.isLeft(p);
}

HOSTDEV auto
isLeftOld(um2::QuadraticSegment<D, T> const & Q, um2::Point<D, T> const & p) -> bool
{
  um2::Vec2<T> const v13 = Q[2] - Q[0];
  um2::Vec2<T> const v23 = Q[2] - Q[1];
  um2::Vec2<T> const va = -2 * (v13 + v23);
  um2::Vec2<T> const vb = 3 * v13 + v23;
  um2::Vec2<T> const vr ( -vb[0] / (2 * va[0]), -vb[1] / (2 * va[1]));
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
  bool const in_box = vmin[0] <= p[0] && p[0] <= vmax[0] &&
                      vmin[1] <= p[1] && p[1] <= vmax[1];
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
      } else {
          return 0 <= (Q[1] - Q[0]).cross(p - Q[0]);
      }
  } else {
      thrust::complex<T> const t1 =
          exp(
              static_cast<T>(0.3333333333333333) * log(
                  static_cast<T>(0.5) * (A + sqrt(static_cast<thrust::complex<T>>(disc)))
              )
          );
      thrust::complex<T> const t2 = t1 == static_cast<thrust::complex<T>>(0) ?
          static_cast<thrust::complex<T>>(0) : B / t1;
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
}

static void isLeftBench(benchmark::State& state) {
//  auto const seg = makeBaseSeg();
  auto const seg = makeSeg4();
  auto const points = makePoints(static_cast<Size>(state.range(0)));
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & p : points) {
     i += static_cast<int>(seg.isLeft(p));
    }
    benchmark::DoNotOptimize(i);
  }
}

static void isLeftOldBench(benchmark::State& state) {
  // auto const seg = makeBaseSeg();
  auto const seg = makeSeg4();
  auto const points = makePoints(static_cast<Size>(state.range(0)));
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & p : points) {
     i += static_cast<int>(isLeftOld(seg, p));
    }
    benchmark::DoNotOptimize(i);
  }
}

BENCHMARK(isLeftBench)->RangeMultiplier(2)->Range(128, NPOINTS)
                          ->Unit(benchmark::kMicrosecond);
BENCHMARK(isLeftOldBench)->RangeMultiplier(2)->Range(128, NPOINTS)
                          ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
