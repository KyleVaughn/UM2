#include "../helpers.hpp"

constexpr Size npoints = 1 << 20;

template <typename T>
static void
crossCPU(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Vec2<T>> const x = makeVectorOfRandomPoints<2, T, -10, 10>(n);
  um2::Vector<um2::Vec2<T>> const y = makeVectorOfRandomPoints<2, T, -10, 10>(n);
  um2::Vector<T> val(n);
  auto const binary_op = [](um2::Vec2<T> const & a, um2::Vec2<T> const & b) {
    return a.cross(b);
  };
  // NOLINTNEXTLINE
  for (auto s : state) {
    std::transform(x.cbegin(), x.cend(), y.cbegin(), val.begin(), binary_op);
  }
}

BENCHMARK_TEMPLATE(crossCPU, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(crossCPU, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();