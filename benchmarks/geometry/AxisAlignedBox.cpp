//=============================================================================
// FINDINGS:
//=============================================================================
// It is not helpful to implement an OpenMP version of the bounding box of a vector
// of points. The single threaded version is faster than the OpenMP version for all
// sizes under very large N

#include "../helpers.hpp"
#include <iostream>

constexpr Size npoints = 1 << 22;
constexpr int lo = -100;
constexpr int hi = 100;

template <typename T>
void
boundingBox(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> box({lo, lo}, {hi, hi});
  um2::Vector<um2::Vec2<T>> const points = makeVectorOfRandomPoints(n, box);
  T xmin = hi;
  T xmax = lo;
  T ymin = hi;
  T ymax = lo;
  for (auto const & p : points) {
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
  for (auto s : state) {
    // cppcheck-suppress useStlAlgorithm; justification: this already uses STL...
    box = um2::boundingBox(points);
  }
  if (!(um2::abs(box.xMin() - xmin) < static_cast<T>(1e-6))) {
    std::cerr << "xMin: " << box.xMin() << " != " << xmin << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(um2::abs(box.xMax() - xmax) < static_cast<T>(1e-6))) {
    std::cerr << "xMax: " << box.xMax() << " != " << xmax << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(um2::abs(box.yMin() - ymin) < static_cast<T>(1e-6))) {
    std::cerr << "yMin: " << box.yMin() << " != " << ymin << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(um2::abs(box.yMax() - ymax) < static_cast<T>(1e-6))) {
    std::cerr << "yMax: " << box.yMax() << " != " << ymax << std::endl;
    exit(EXIT_FAILURE);
  }
}

BENCHMARK_TEMPLATE(boundingBox, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
