//=====================================================================
// Findings
//=====================================================================

#include "../helpers.hpp"

#include <um2/visualization/Image2D.hpp>

#define T float

constexpr Size nlines = 1 << 14;
constexpr Size lo = 0;
constexpr Size hi = 1024;

class ImageFixture : public benchmark::Fixture
{
public:
  um2::Image2D<T> image;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    image.grid.minima[0] = static_cast<T>(0);
    image.grid.minima[1] = static_cast<T>(0);
    image.grid.spacing[0] = static_cast<T>(1);
    image.grid.spacing[1] = static_cast<T>(1);
    image.grid.num_cells[0] = hi;
    image.grid.num_cells[1] = hi;
    image.children.resize(hi * hi);
  }

  void
  TearDown(const ::benchmark::State & /*state*/) final
  {
  }
};

// cppcheck-suppress unknownMacro
BENCHMARK_DEFINE_F(ImageFixture, rasterize)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<T> const box({lo, lo}, {hi, hi});
  auto const triangles = makeVectorOfRandomTriangles(n,box);

  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & tri : triangles) {
      image.rasterize(tri);
    }
    state.PauseTiming();
    image.clear();
    state.ResumeTiming();
  }
}

BENCHMARK_REGISTER_F(ImageFixture, rasterize)
    ->RangeMultiplier(4)
    ->Range(1024, nlines)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();