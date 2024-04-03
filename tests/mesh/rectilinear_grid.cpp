#include <um2/mesh/rectilinear_grid.hpp>

#include "../test_macros.hpp"

Float constexpr eps = castIfNot<Float>(1e-6);

template <Int D>
HOSTDEV constexpr auto
makeGrid() -> um2::RectilinearGrid<D>
{
  um2::RectilinearGrid<D> grid;
  if constexpr (D >= 1) {
    grid.divs(0) = {0, 1};
  }
  if constexpr (D >= 2) {
    grid.divs(1) = {0, 1, 2};
  }
  if constexpr (D >= 3) {
    grid.divs(2) = {0, 1, 2, 3};
  }
  return grid;
}

template <Int D>
HOSTDEV
TEST_CASE(clear)
{
  um2::RectilinearGrid<D> grid = makeGrid<D>();
  grid.clear();
  for (Int i = 0; i < D; ++i) {
    ASSERT(grid.divs(i).empty());
  }
}

template <Int D>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RectilinearGrid<D> const grid = makeGrid<D>();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(grid.minima(i), 0, eps);
    ASSERT_NEAR(grid.maxima(i), static_cast<Float>(i + 1), eps);
    ASSERT(grid.numCells(i) == i + 1);
    ASSERT_NEAR(grid.extents(i), static_cast<Float>(i + 1), eps);
  }
}

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RectilinearGrid<D> const grid = makeGrid<D>();
  um2::AxisAlignedBox<D> const box = grid.boundingBox();
  ASSERT(box.minima().isApprox(um2::Point<D>::zero()));
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box.maxima(i), static_cast<Float>(i + 1), eps);
  }
}

HOSTDEV
TEST_CASE(getBox)
{
  // Declare some variables to avoid a bunch of static casts.
  auto const three = static_cast<Float>(3);
  auto const two = static_cast<Float>(2);
  auto const one = static_cast<Float>(1);
  auto const half = static_cast<Float>(1) / static_cast<Float>(2);
  auto const forth = static_cast<Float>(1) / static_cast<Float>(4);
  um2::RectilinearGrid2 grid;
  grid.divs(0) = {1.0, 1.5, 2.0, 2.5, 3.0};
  grid.divs(1) = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
  um2::AxisAlignedBox2 box = grid.getBox(0, 0);
  um2::AxisAlignedBox2 box_ref = {
      {         1,             -1},
      {one + half, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + half,           -one},
      {       two, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + half,           -one},
      {     three, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {       one, -three * forth},
      {one + half,          -half}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {       one, three * forth},
      {one + half,           one}
  };
  ASSERT(box.isApprox(box_ref));
  box = grid.getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + half, three * forth},
      {     three,           one}
  };
  ASSERT(box.isApprox(box_ref));
}

TEST_CASE(aabb_constructor)
{
  um2::AxisAlignedBox2 const b00(um2::Point2(0, 0), um2::Point2(1, 1));
  um2::AxisAlignedBox2 const b10(um2::Point2(1, 0), um2::Point2(2, 1));
  um2::AxisAlignedBox2 const b01(um2::Point2(0, 1), um2::Point2(1, 2));
  um2::AxisAlignedBox2 const b11(um2::Point2(1, 1), um2::Point2(2, 2));
  um2::AxisAlignedBox2 const b02(um2::Point2(0, 2), um2::Point2(1, 3));
  um2::AxisAlignedBox2 const b12(um2::Point2(1, 2), um2::Point2(2, 3));
  um2::Vector<um2::AxisAlignedBox2> const boxes = {b00, b10, b01, b11, b02, b12};
  um2::RectilinearGrid2 grid(boxes);

  ASSERT(grid.divs(0).size() == 3);
  Float const xref[3] = {0, 1, 2};
  for (Int i = 0; i < 3; ++i) {
    ASSERT_NEAR(grid.divs(0)[i], xref[i], eps);
  }

  ASSERT(grid.divs(1).size() == 4);
  Float const yref[4] = {0, 1, 2, 3};
  for (Int i = 0; i < 4; ++i) {
    ASSERT_NEAR(grid.divs(1)[i], yref[i], eps);
  }

  um2::RectilinearGrid2 grid2(b01);
  ASSERT(grid2.divs(0).size() == 2);
  ASSERT(grid2.divs(1).size() == 2);
  ASSERT_NEAR(grid2.divs(0)[0], 0, eps);
  ASSERT_NEAR(grid2.divs(0)[1], 1, eps);
  ASSERT_NEAR(grid2.divs(1)[0], 1, eps);
  ASSERT_NEAR(grid2.divs(1)[1], 2, eps);
}

TEST_CASE(id_array_constructor)
{
  um2::Vector<um2::Vector<Int>> const ids = {
      {0, 1, 2, 0},
      {0, 2, 0, 2},
      {0, 1, 0, 1},
  };
  um2::Vector<um2::Vec2<Float>> const dxdy = {
      {2, 1},
      {2, 1},
      {2, 1},
      {2, 1},
  };
  um2::RectilinearGrid2 grid(dxdy, ids);

  ASSERT(grid.divs(0).size() == 5);
  Float const xref[5] = {0, 2, 4, 6, 8};
  for (Int i = 0; i < 5; ++i) {
    ASSERT_NEAR(grid.divs(0)[i], xref[i], eps);
  }
  ASSERT(grid.divs(1).size() == 4);
  Float const yref[4] = {0, 1, 2, 3};
  for (Int i = 0; i < 4; ++i) {
    ASSERT_NEAR(grid.divs(1)[i], yref[i], eps);
  }
}

template <Int D>
TEST_SUITE(RectilinearGrid)
{
  TEST_HOSTDEV(clear, D);
  TEST_HOSTDEV(accessors, D);
  TEST_HOSTDEV(boundingBox, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox);
    TEST(aabb_constructor);
    TEST(id_array_constructor);
  }
}

auto
main() -> int
{
  RUN_SUITE(RectilinearGrid<1>);
  RUN_SUITE(RectilinearGrid<2>);
  RUN_SUITE(RectilinearGrid<3>);
  return 0;
}
