#include <um2/mesh/regular_grid.hpp>

#include "../test_macros.hpp"

Float constexpr eps = castIfNot<Float>(1e-6);

template <Int D>
HOSTDEV constexpr auto
makeGrid() -> um2::RegularGrid<D>
{
  static_assert(1 <= D && D <= 3, "D must be in [1, 3]");
  um2::Point<D> minima;
  um2::Point<D> spacing;
  um2::Vec<D, Int> num_cells;
  for (Int i = 0; i < D; ++i) {
    minima[i] = castIfNot<Float>(i + 1);
    spacing[i] = castIfNot<Float>(i + 1);
    num_cells[i] = i + 1;
  }
  return {minima, spacing, num_cells};
}

template <Int D>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RegularGrid<D> const grid = makeGrid<D>();
  if constexpr (D >= 1) {
    Float const xmin = grid.minima()[0];
    ASSERT_NEAR(grid.xMin(), xmin, eps);
    Float const dx = grid.spacing()[0];
    ASSERT_NEAR(grid.dx(), dx, eps);
    auto const nx = grid.numCells()[0];
    ASSERT(grid.numXCells() == nx);
    ASSERT_NEAR(grid.width(), dx * castIfNot<Float>(nx), eps);
    ASSERT_NEAR(grid.xMax(), xmin + dx * castIfNot<Float>(nx), eps);
  }
  if constexpr (D >= 2) {
    Float const ymin = grid.minima()[1];
    ASSERT_NEAR(grid.yMin(), ymin, eps);
    Float const dy = grid.spacing()[1];
    ASSERT_NEAR(grid.dy(), dy, eps);
    auto const ny = grid.numCells()[1];
    ASSERT(grid.numYCells() == ny);
    ASSERT_NEAR(grid.height(), dy * castIfNot<Float>(ny), eps);
    ASSERT_NEAR(grid.yMax(), ymin + dy * castIfNot<Float>(ny), eps);
  }
  if constexpr (D >= 3) {
    Float const zmin = grid.minima()[2];
    ASSERT_NEAR(grid.zMin(), zmin, eps);
    Float const dz = grid.spacing()[2];
    ASSERT_NEAR(grid.dz(), dz, eps);
    auto const nz = grid.numCells()[2];
    ASSERT(grid.numZCells() == nz);
    ASSERT_NEAR(grid.depth(), dz * castIfNot<Float>(nz), eps);
    ASSERT_NEAR(grid.zMax(), zmin + dz * castIfNot<Float>(nz), eps);
  }
}

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RegularGrid<D> const grid = makeGrid<D>();
  um2::AxisAlignedBox<D> const box = grid.boundingBox();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.minima()[0], grid.xMin(), eps);
    ASSERT_NEAR(box.maxima()[0], grid.xMax(), eps);
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.minima()[1], grid.yMin(), eps);
    ASSERT_NEAR(box.maxima()[1], grid.yMax(), eps);
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.minima()[2], grid.zMin(), eps);
    ASSERT_NEAR(box.maxima()[2], grid.zMax(), eps);
  }
}

HOSTDEV
TEST_CASE(getBox)
{
  // Declare some variables to avoid a bunch of static casts.
  auto const three = castIfNot<Float>(3);
  auto const two = castIfNot<Float>(2);
  auto const one = castIfNot<Float>(1);
  auto const ahalf = castIfNot<Float>(1) / castIfNot<Float>(2);
  auto const forth = castIfNot<Float>(1) / castIfNot<Float>(4);
  um2::Point2 const minima = {1, -1};
  um2::Vec2<Float> const spacing = {ahalf, forth};
  um2::Vec2<Int> const num_cells = {4, 8};
  um2::RegularGrid2 const grid(minima, spacing, num_cells);
  um2::AxisAlignedBox2 box = grid.getBox(0, 0);
  um2::AxisAlignedBox2 box_ref = {
      {          1,             -1},
      {one + ahalf, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = grid.getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + ahalf,           -one},
      {        two, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = grid.getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + ahalf,           -one},
      {      three, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = grid.getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {        one, -three * forth},
      {one + ahalf,         -ahalf}
  };
  ASSERT(isApprox(box, box_ref));
  box = grid.getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {        one, three * forth},
      {one + ahalf,           one}
  };
  ASSERT(isApprox(box, box_ref));
  box = grid.getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + ahalf, three * forth},
      {      three,           one}
  };
  ASSERT(isApprox(box, box_ref));
}

template <Int D>
HOSTDEV
TEST_CASE(getCellCentroid)
{
  um2::RegularGrid<D> const grid = makeGrid<D>();
  if constexpr (D == 1) {
    auto const x = grid.getCellCentroid(0);
    ASSERT_NEAR(x[0], grid.minima()[0] + grid.spacing()[0] / castIfNot<Float>(2), eps);
  }
  if constexpr (D == 2) {
    auto const y = grid.getCellCentroid(0, 0);
    ASSERT_NEAR(y[1], grid.minima()[1] + grid.spacing()[1] / castIfNot<Float>(2), eps);
  }
}

HOSTDEV
TEST_CASE(getCellIndicesIntersecting)
{
  um2::Point2 const minima(1, -1);
  um2::Vec2<Float> const spacing(2, 1);
  um2::Vec2<Int> const num_cells(5, 8);
  // Grid ranges from 1 to 11 in x and -1 to 7 in y.
  um2::RegularGrid2 const grid(minima, spacing, num_cells);

  // A box in a single cell.
  um2::AxisAlignedBox2 const box0({castIfNot<Float>(3.1), castIfNot<Float>(1.1)},
                                  {castIfNot<Float>(3.9), castIfNot<Float>(1.9)});
  um2::Vec<4, Int> const range0 = grid.getCellIndicesIntersecting(box0);
  ASSERT(range0[0] == 1);
  ASSERT(range0[1] == 2);
  ASSERT(range0[2] == 1);
  ASSERT(range0[3] == 2);

  // A box with perfect alignment.
  um2::AxisAlignedBox2 const box1({castIfNot<Float>(3), castIfNot<Float>(1)},
                                  {castIfNot<Float>(5), castIfNot<Float>(2)});
  um2::Vec<4, Int> const range1 = grid.getCellIndicesIntersecting(box1);
  ASSERT(range1[0] == 0 || range1[0] == 1);
  ASSERT(range1[1] == 1 || range1[1] == 2);
  ASSERT(range1[2] == 1 || range1[2] == 2); // Valid in either cell.
  ASSERT(range1[3] == 2 || range1[3] == 3);

  // A box in multiple cells.
  um2::AxisAlignedBox2 const box2({castIfNot<Float>(3.1), castIfNot<Float>(1.1)},
                                  {castIfNot<Float>(5.9), castIfNot<Float>(1.9)});
  um2::Vec<4, Int> const range2 = grid.getCellIndicesIntersecting(box2);
  ASSERT(range2[0] == 1);
  ASSERT(range2[1] == 2);
  ASSERT(range2[2] == 2);
  ASSERT(range2[3] == 2);

  // A box in 4 cells.
  um2::AxisAlignedBox2 const box3({castIfNot<Float>(3.1), castIfNot<Float>(1.1)},
                                  {castIfNot<Float>(5.9), castIfNot<Float>(2.9)});
  um2::Vec<4, Int> const range3 = grid.getCellIndicesIntersecting(box3);
  ASSERT(range3[0] == 1);
  ASSERT(range3[1] == 2);
  ASSERT(range3[2] == 2);
  ASSERT(range3[3] == 3);
}

HOSTDEV
TEST_CASE(getCellIndexContaining)
{
  um2::Point2 const minima(1, -1);
  um2::Vec2<Float> const spacing(2, 1);
  um2::Vec2<Int> const num_cells(5, 8);
  // Grid ranges from 1 to 11 in x and -1 to 7 in y.
  um2::RegularGrid2 const grid(minima, spacing, num_cells);
  um2::Vec<2, Int> id = grid.getCellIndexContaining({castIfNot<Float>(1.1), castIfNot<Float>(1.1)});
  ASSERT(id[0] == 0);
  ASSERT(id[1] == 2);
  id = grid.getCellIndexContaining({castIfNot<Float>(4.9), castIfNot<Float>(2.1)});
  ASSERT(id[0] == 1);
  ASSERT(id[1] == 3);
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(constructor, D)

template <Int D>
MAKE_CUDA_KERNEL(accessors, D)

template <Int D>
MAKE_CUDA_KERNEL(boundingBox, D)

template <Int D>
MAKE_CUDA_KERNEL(getCellCentroid, D)

MAKE_CUDA_KERNEL(getBox)

    MAKE_CUDA_KERNEL(getCellIndicesIntersecting)

        MAKE_CUDA_KERNEL(getCellIndexContaining)
#endif

            template <Int D>
            TEST_SUITE(RegularGrid)
{
  TEST_HOSTDEV(accessors, D);
  TEST_HOSTDEV(boundingBox, D);
  TEST_HOSTDEV(getCellCentroid, D)
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox);
    TEST_HOSTDEV(getCellIndicesIntersecting);
    TEST_HOSTDEV(getCellIndexContaining);
  }
}

auto
main() -> int
{
  RUN_SUITE(RegularGrid<1>);
  RUN_SUITE(RegularGrid<2>);
  RUN_SUITE(RegularGrid<3>);
  return 0;
}
