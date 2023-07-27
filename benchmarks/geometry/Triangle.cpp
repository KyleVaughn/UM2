// FINDINGS: The barycentric method is faster than the CCW method on some processors, but
// not others

#include <benchmark/benchmark.h>
#include <um2/geometry/Triangle.hpp>

#include <random>

// NOLINTBEGIN
#define D       2
#define T       float
#define NPOINTS 1 << 16
#define NTRIS   1 << 3
#define BOX_MAX 10
// NOLINTEND

auto
randomPoint() -> um2::Point<D, T>
{
  // NOLINTNEXTLINE
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(0, BOX_MAX);
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = dist(rng);
  }
  return p;
}

// NOLINTBEGIN(readability-identifier-naming)
constexpr auto
triContainsBary(um2::Triangle<D, T> const & tri, um2::Point<D, T> const & p) -> bool
{
  um2::Vec<D, T> const A = tri[1] - tri[0];
  um2::Vec<D, T> const B = tri[2] - tri[0];
  um2::Vec<D, T> const C = p - tri[0];
  T const invdetAB = 1 / A.cross(B);
  T const r = C.cross(B) * invdetAB;
  T const s = A.cross(C) * invdetAB;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
}

constexpr auto
triContainsCCW(um2::Triangle<D, T> const & tri, um2::Point<D, T> const & p) -> bool
{
  return um2::areCCW(tri[0], tri[1], p) && um2::areCCW(tri[1], tri[2], p) &&
         um2::areCCW(tri[2], tri[0], p);
}
// NOLINTEND(readability-identifier-naming)

static void
containsBary(benchmark::State & state)
{
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  um2::Vector<um2::Triangle<D, T>> tris(NTRIS);
  for (auto & p : points) {
    // cppcheck-suppress useStlAlgorithm
    p = randomPoint();
  }
  for (auto & t : tris) {
    // cppcheck-suppress useStlAlgorithm
    t = um2::Triangle<D, T>(randomPoint(), randomPoint(), randomPoint());
  }
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & t : tris) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(triContainsBary(t, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

static void
containsCCW(benchmark::State & state)
{
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  um2::Vector<um2::Triangle<D, T>> tris(NTRIS);
  for (auto & p : points) {
    // cppcheck-suppress useStlAlgorithm
    p = randomPoint();
  }
  for (auto & t : tris) {
    // cppcheck-suppress useStlAlgorithm
    t = um2::Triangle<D, T>(randomPoint(), randomPoint(), randomPoint());
  }
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & t : tris) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(triContainsCCW(t, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

BENCHMARK(containsBary)->RangeMultiplier(2)->Range(16, NPOINTS);
BENCHMARK(containsCCW)->RangeMultiplier(2)->Range(16, NPOINTS);
BENCHMARK_MAIN();
