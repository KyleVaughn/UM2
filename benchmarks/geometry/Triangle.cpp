//=====================================================================
// Findings
//=====================================================================
// For 65536 points and 1024 triangles:
// Barycentric:   27695 us = 0.41 ns/point
// CCW:          307249 us = 4.58 ns/point
// CCW no short:  33773 us = 0.50 ns/point
//
// Although CCW no short is slower than Barycentric, in these results, benchmarking
// on multiple processors shows that CCW no short is faster on average.

#include "../helpers.hpp"
#include <um2/geometry/Triangle.hpp>

constexpr Size npoints = 1 << 16;
constexpr Size ntris = 1 << 10;
constexpr int lo = 0;
constexpr int hi = 100;

template <typename T>
constexpr auto
triContainsBary(um2::Triangle2<T> const & tri, um2::Point2<T> const & p) -> bool
{
  um2::Vec2<T> const a = tri[1] - tri[0];
  um2::Vec2<T> const b = tri[2] - tri[0];
  um2::Vec2<T> const c = p - tri[0];
  T const invdetAB = 1 / a.cross(b);
  T const r = c.cross(b) * invdetAB;
  T const s = a.cross(c) * invdetAB;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
}

template <typename T>
constexpr auto
triContainsCCW(um2::Triangle2<T> const & tri, um2::Point2<T> const & p) -> bool
{
  return um2::areCCW(tri[0], tri[1], p) && um2::areCCW(tri[1], tri[2], p) &&
         um2::areCCW(tri[2], tri[0], p);
}

template <typename T>
constexpr auto
triContainsNoShortCCW(um2::Triangle2<T> const & tri, um2::Point2<T> const & p) -> bool
{
  bool const b0 = um2::areCCW(tri[0], tri[1], p);
  bool const b1 = um2::areCCW(tri[1], tri[2], p);
  bool const b2 = um2::areCCW(tri[2], tri[0], p);
  return b0 && b1 && b2;
}

template <typename T>
static void
containsBary(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  auto const tris = makeVectorOfRandomTriangles<T>(ntris, box);
  for (auto s : state) {
    int64_t i = 0;
    for (auto const & t : tris) {
      i += std::count_if(points.begin(), points.end(),
                         [&t](auto const & p) { return triContainsBary(t, p); });
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
static void
containsCCW(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  auto const tris = makeVectorOfRandomTriangles<T>(ntris, box);
  for (auto s : state) {
    int64_t i = 0;
    for (auto const & t : tris) {
      i += std::count_if(points.begin(), points.end(),
                         [&t](auto const & p) { return triContainsCCW(t, p); });
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
static void
containsNoShortCCW(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  auto const tris = makeVectorOfRandomTriangles<T>(ntris, box);
  for (auto s : state) {
    int64_t i = 0;
    for (auto const & t : tris) {
      i += std::count_if(points.begin(), points.end(),
                         [&t](auto const & p) { return triContainsNoShortCCW(t, p); });
    }
    benchmark::DoNotOptimize(i);
  }
}

BENCHMARK_TEMPLATE(containsBary, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCW, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsNoShortCCW, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
