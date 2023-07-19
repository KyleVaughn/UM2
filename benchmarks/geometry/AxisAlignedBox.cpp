#include <benchmark/benchmark.h>
#include <um2/geometry/AxisAlignedBox.hpp>

#include <random>

#define D 2
#define T float 
#define NPOINTS 1 << 20 
#define BOX_MAX 1000

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

static void boundingBox(benchmark::State& state) {
  um2::Vector<um2::Point<D, T>> points(static_cast<Size>(state.range(0)));
  T xmin = BOX_MAX;
  T xmax = 0;
  T ymin = BOX_MAX;
  T ymax = 0;
  for (auto& p : points) {
    p = randomPoint();
    if (p[0] < xmin) {
      xmin = p[0];
    } 
    if (p[0] > xmax) {
      xmax = p[0];
    } 
    if (p[1] < ymin) {
      ymin = p[1];
    }
    if (p[1] > ymax) {
      ymax = p[1];
    }
  }
  um2::AxisAlignedBox<D, T> box;
  // NOLINTNEXTLINE
  for (auto s : state) {
    box = um2::boundingBox(points);
    benchmark::DoNotOptimize(box);
    benchmark::ClobberMemory();
  }
  assert(um2::abs(box.xMin() - xmin) < static_cast<T>(1e-6));
  assert(um2::abs(box.xMax() - xmax) < static_cast<T>(1e-6));
  assert(um2::abs(box.yMin() - ymin) < static_cast<T>(1e-6));
  assert(um2::abs(box.yMax() - ymax) < static_cast<T>(1e-6));
}

BENCHMARK(boundingBox)->RangeMultiplier(2)->Range(16, NPOINTS);
BENCHMARK_MAIN();
