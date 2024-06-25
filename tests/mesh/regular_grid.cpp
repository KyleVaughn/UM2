#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/math/vec.hpp>
#include <um2/mesh/regular_grid.hpp>

#include "../test_macros.hpp"

template <class T>
T constexpr eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makeGrid() -> um2::RegularGrid<D, T>
{
  static_assert(1 <= D && D <= 3, "D must be in [1, 3]");
  um2::Point<D, T> minima;
  um2::Point<D, T> spacing;
  um2::Vec<D, Int> num_cells;
  for (Int i = 0; i < D; ++i) {
    minima[i] = castIfNot<T>(i + 1);
    spacing[i] = castIfNot<T>(i + 1);
    num_cells[i] = i + 1;
  }
  return {minima, spacing, num_cells};
}

CONST HOSTDEV auto
factorial(Int n) -> Int
{
  Int result = 1;
  for (Int i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}

template <Int D, class T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RegularGrid<D, T> const grid = makeGrid<D, T>();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(grid.minima(i), castIfNot<T>(i + 1), eps<T>);
    ASSERT_NEAR(grid.spacing(i), castIfNot<T>(i + 1), eps<T>);
    ASSERT(grid.numCells(i) == i + 1);
  }
  ASSERT(grid.totalNumCells() == factorial(D));

  for (Int i = 0; i < D; ++i) {
    auto const i_1 = castIfNot<T>(i + 1);
    auto const i_1_sq = i_1 * i_1;
    ASSERT_NEAR(grid.extents(i), i_1_sq, eps<T>);
    ASSERT_NEAR(grid.maxima(i), i_1_sq + i_1, eps<T>);
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RegularGrid<D, T> const grid = makeGrid<D, T>();
  um2::AxisAlignedBox<D, T> const box = grid.boundingBox();
  ASSERT(box.minima().isApprox(grid.minima()));
  ASSERT(box.maxima().isApprox(grid.maxima()));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(getCellCentroid)
{
  um2::RegularGrid<D, T> const grid = makeGrid<D, T>();
  if constexpr (D == 1) {
    auto const x = grid.getCellCentroid(0);
    ASSERT_NEAR(x[0], grid.minima(0) + grid.spacing(0) / castIfNot<T>(2), eps<T>);
  }
  if constexpr (D == 2) {
    auto const xy = grid.getCellCentroid(0, 0);
    ASSERT_NEAR(xy[1], grid.minima(1) + grid.spacing(1) / castIfNot<T>(2), eps<T>);
  }
}

template <class T>
HOSTDEV
TEST_CASE(getBox)
{
  // Declare some variables to avoid a bunch of static casts.
  auto const three = castIfNot<T>(3);
  auto const two = castIfNot<T>(2);
  auto const one = castIfNot<T>(1);
  auto const ahalf = castIfNot<T>(1) / castIfNot<T>(2);
  auto const forth = castIfNot<T>(1) / castIfNot<T>(4);
  um2::Point2<T> const minima = {1, -1};
  um2::Vec2<T> const spacing = {ahalf, forth};
  um2::Vec2<Int> const num_cells = {4, 8};
  um2::RegularGrid2<T> const grid(minima, spacing, num_cells);
  um2::AxisAlignedBox2<T> box = grid.getBox(0, 0);
  um2::AxisAlignedBox2<T> box_ref = {
      {          1,             -1},
      {one + ahalf, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + ahalf,           -one},
      {        two, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + ahalf,           -one},
      {      three, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {        one, -three * forth},
      {one + ahalf,         -ahalf}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {        one, three * forth},
      {one + ahalf,           one}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + ahalf, three * forth},
      {      three,           one}
  };
  ASSERT(box.isApprox(box_ref));
}

template <class T>
HOSTDEV
TEST_CASE(getCellIndicesIntersecting)
{
  um2::Point2<T> const minima(1, -1);
  um2::Vec2<T> const spacing(2, 1);
  um2::Vec2<Int> const num_cells(5, 8);
  // Grid ranges from 1 to 11 in x and -1 to 7 in y.
  um2::RegularGrid2<T> const grid(minima, spacing, num_cells);

  // A box in a single cell.
  um2::AxisAlignedBox2<T> const box0({castIfNot<T>(3.1), castIfNot<T>(1.1)},
                                     {castIfNot<T>(3.9), castIfNot<T>(1.9)});
  um2::Vec<4, Int> const range0 = grid.getCellIndicesIntersecting(box0);
  ASSERT(range0[0] == 1);
  ASSERT(range0[1] == 2);
  ASSERT(range0[2] == 1);
  ASSERT(range0[3] == 2);

  // A box with perfect alignment.
  um2::AxisAlignedBox2<T> const box1({castIfNot<T>(3), castIfNot<T>(1)},
                                     {castIfNot<T>(5), castIfNot<T>(2)});
  um2::Vec<4, Int> const range1 = grid.getCellIndicesIntersecting(box1);
  ASSERT(range1[0] == 0 || range1[0] == 1);
  ASSERT(range1[1] == 1 || range1[1] == 2);
  ASSERT(range1[2] == 1 || range1[2] == 2); // Valid in either cell.
  ASSERT(range1[3] == 2 || range1[3] == 3);

  // A box in multiple cells.
  um2::AxisAlignedBox2<T> const box2({castIfNot<T>(3.1), castIfNot<T>(1.1)},
                                     {castIfNot<T>(5.9), castIfNot<T>(1.9)});
  um2::Vec<4, Int> const range2 = grid.getCellIndicesIntersecting(box2);
  ASSERT(range2[0] == 1);
  ASSERT(range2[1] == 2);
  ASSERT(range2[2] == 2);
  ASSERT(range2[3] == 2);

  // A box in 4 cells.
  um2::AxisAlignedBox2<T> const box3({castIfNot<T>(3.1), castIfNot<T>(1.1)},
                                     {castIfNot<T>(5.9), castIfNot<T>(2.9)});
  um2::Vec<4, Int> const range3 = grid.getCellIndicesIntersecting(box3);
  ASSERT(range3[0] == 1);
  ASSERT(range3[1] == 2);
  ASSERT(range3[2] == 2);
  ASSERT(range3[3] == 3);
}

template <class T>
HOSTDEV
TEST_CASE(getCellIndexContaining)
{
  um2::Point2<T> const minima(1, -1);
  um2::Vec2<T> const spacing(2, 1);
  um2::Vec2<Int> const num_cells(5, 8);
  // Grid ranges from 1 to 11 in x and -1 to 7 in y.
  um2::RegularGrid2<T> const grid(minima, spacing, num_cells);
  um2::Vec<2, Int> id =
      grid.getCellIndexContaining({castIfNot<T>(1.1), castIfNot<T>(1.1)});
  ASSERT(id[0] == 0);
  ASSERT(id[1] == 2);
  id = grid.getCellIndexContaining({castIfNot<T>(4.9), castIfNot<T>(2.1)});
  ASSERT(id[0] == 1);
  ASSERT(id[1] == 3);
}

template <Int D, class T>
TEST_SUITE(RegularGrid)
{
  TEST_HOSTDEV(accessors, D, T);
  TEST_HOSTDEV(boundingBox, D, T);
  TEST_HOSTDEV(getCellCentroid, D, T)
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox, T);
    TEST_HOSTDEV(getCellIndicesIntersecting, T);
    TEST_HOSTDEV(getCellIndexContaining, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((RegularGrid<1, float>));
  RUN_SUITE((RegularGrid<2, float>));
  RUN_SUITE((RegularGrid<3, float>));

  RUN_SUITE((RegularGrid<1, double>));
  RUN_SUITE((RegularGrid<2, double>));
  RUN_SUITE((RegularGrid<3, double>));
  return 0;
}
