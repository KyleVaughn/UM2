#include <benchmark/benchmark.h>
#include <um2/geometry/Triangle.hpp>
#include <um2/geometry/Quadrilateral.hpp>

#include <random>

#define D 2
#define T float 
#define NPOINTS 1 << 16 
#define BOX_MAX 10

auto randomPoint() -> um2::Point<D, T> 
{
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
isConvexArea(um2::Quadrilateral<D, T> const & quad) -> bool
{
  T const A1 = um2::Triangle<D, T>(quad[0], quad[1], quad[2]).area();
  T const A2 = um2::Triangle<D, T>(quad[2], quad[3], quad[0]).area();
  T const A3 = um2::Triangle<D, T>(quad[3], quad[0], quad[1]).area();
  T const A4 = um2::Triangle<D, T>(quad[1], quad[2], quad[3]).area();
  return um2::abs(A1 + A2 - A3 - A4) < static_cast<T>(0.01) * (A1 + A2); 
}

constexpr auto
isConvexCCW(um2::Quadrilateral<D, T> const & quad) -> bool
{
  return um2::areCCW(quad[0], quad[1], quad[2]) &&
         um2::areCCW(quad[1], quad[2], quad[3]) &&
         um2::areCCW(quad[2], quad[3], quad[0]) &&
         um2::areCCW(quad[3], quad[0], quad[1]);
}
// NOLINTEND(readability-identifier-naming)

static void isConvexArea(benchmark::State& state) {
  um2::Vector<um2::Quadrilateral<D, T>> quads(static_cast<Size>(state.range(0)));
  for (auto& q : quads) {
    auto p0 = randomPoint();
    auto p1 = p0;
    p1[0] += 1;
    auto p2 = p0;
    p2[0] += 1;
    p2[1] += 1;
    q = um2::Quadrilateral<D, T>(p0, p1, p2, randomPoint());
  }
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
        i += static_cast<int>(isConvexArea(q));
    }
    benchmark::DoNotOptimize(i);
  }
}

static void isConvexCCW(benchmark::State& state) {
  um2::Vector<um2::Quadrilateral<D, T>> quads(static_cast<Size>(state.range(0)));
  for (auto& q : quads) {
    // cppcheck-suppress useStlAlgorithm
    q = um2::Quadrilateral<D, T>(randomPoint(), randomPoint(), randomPoint(), randomPoint());
  }
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
        i += static_cast<int>(isConvexCCW(q));
    }
    benchmark::DoNotOptimize(i);
  }
}

BENCHMARK(isConvexArea)->RangeMultiplier(2)->Range(16, NPOINTS);
BENCHMARK(isConvexCCW)->RangeMultiplier(2)->Range(16, NPOINTS);
BENCHMARK_MAIN();
