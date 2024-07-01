#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/math/vec.hpp>
#include <um2/mesh/rectilinear_grid.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

template <class T>
constexpr T eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makeGrid() -> um2::RectilinearGrid<D, T>
{
  um2::RectilinearGrid<D, T> grid;
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

template <Int D, class T>
HOSTDEV
TEST_CASE(clear)
{
  um2::RectilinearGrid<D, T> grid = makeGrid<D, T>();
  grid.clear();
  for (Int i = 0; i < D; ++i) {
    ASSERT(grid.divs(i).empty());
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RectilinearGrid<D, T> const grid = makeGrid<D, T>();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(grid.minima(i), 0, eps<T>);
    ASSERT_NEAR(grid.maxima(i), static_cast<T>(i + 1), eps<T>);
    ASSERT(grid.numCells(i) == i + 1);
    ASSERT_NEAR(grid.extents(i), static_cast<T>(i + 1), eps<T>);
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RectilinearGrid<D, T> const grid = makeGrid<D, T>();
  um2::AxisAlignedBox<D, T> const box = grid.boundingBox();
  ASSERT(box.minima().isApprox(um2::Point<D, T>::zero()));
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box.maxima(i), static_cast<T>(i + 1), eps<T>);
  }
}

template <class T>
HOSTDEV
TEST_CASE(getBox)
{
  // Declare some variables to avoid a bunch of static casts.
  auto const three = static_cast<T>(3);
  auto const two = static_cast<T>(2);
  auto const one = static_cast<T>(1);
  auto const half = static_cast<T>(1) / static_cast<T>(2);
  auto const forth = static_cast<T>(1) / static_cast<T>(4);
  um2::RectilinearGrid2<T> grid;
  grid.divs(0) = {1.0, 1.5, 2.0, 2.5, 3.0};
  grid.divs(1) = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
  um2::AxisAlignedBox2<T> box = grid.getBox(0, 0);
  um2::AxisAlignedBox2<T> box_ref = {
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

template <class T>
TEST_CASE(aabb_constructor)
{
  um2::AxisAlignedBox2<T> const b00(um2::Point2<T>(0, 0), um2::Point2<T>(1, 1));
  um2::AxisAlignedBox2<T> const b10(um2::Point2<T>(1, 0), um2::Point2<T>(2, 1));
  um2::AxisAlignedBox2<T> const b01(um2::Point2<T>(0, 1), um2::Point2<T>(1, 2));
  um2::AxisAlignedBox2<T> const b11(um2::Point2<T>(1, 1), um2::Point2<T>(2, 2));
  um2::AxisAlignedBox2<T> const b02(um2::Point2<T>(0, 2), um2::Point2<T>(1, 3));
  um2::AxisAlignedBox2<T> const b12(um2::Point2<T>(1, 2), um2::Point2<T>(2, 3));
  um2::Vector<um2::AxisAlignedBox2<T>> const boxes = {b00, b10, b01, b11, b02, b12};
  um2::RectilinearGrid2<T> grid(boxes);

  ASSERT(grid.divs(0).size() == 3);
  T const xref[3] = {0, 1, 2};
  for (Int i = 0; i < 3; ++i) {
    ASSERT_NEAR(grid.divs(0)[i], xref[i], eps<T>);
  }

  ASSERT(grid.divs(1).size() == 4);
  T const yref[4] = {0, 1, 2, 3};
  for (Int i = 0; i < 4; ++i) {
    ASSERT_NEAR(grid.divs(1)[i], yref[i], eps<T>);
  }

  um2::RectilinearGrid2<T> grid2(b01);
  ASSERT(grid2.divs(0).size() == 2);
  ASSERT(grid2.divs(1).size() == 2);
  ASSERT_NEAR(grid2.divs(0)[0], 0, eps<T>);
  ASSERT_NEAR(grid2.divs(0)[1], 1, eps<T>);
  ASSERT_NEAR(grid2.divs(1)[0], 1, eps<T>);
  ASSERT_NEAR(grid2.divs(1)[1], 2, eps<T>);
}

template <class T>
TEST_CASE(id_array_constructor)
{
  um2::Vector<um2::Vector<Int>> const ids = {
      {0, 1, 2, 0},
      {0, 2, 0, 2},
      {0, 1, 0, 1},
  };
  um2::Vector<um2::Vec2<T>> const dxdy = {
      {2, 1},
      {2, 1},
      {2, 1},
      {2, 1},
  };
  um2::RectilinearGrid2<T> grid(dxdy, ids);

  ASSERT(grid.divs(0).size() == 5);
  T const xref[5] = {0, 2, 4, 6, 8};
  for (Int i = 0; i < 5; ++i) {
    ASSERT_NEAR(grid.divs(0)[i], xref[i], eps<T>);
  }
  ASSERT(grid.divs(1).size() == 4);
  T const yref[4] = {0, 1, 2, 3};
  for (Int i = 0; i < 4; ++i) {
    ASSERT_NEAR(grid.divs(1)[i], yref[i], eps<T>);
  }
}

template <Int D, class T>
TEST_SUITE(RectilinearGrid)
{
  TEST_HOSTDEV(clear, D, T);
  TEST_HOSTDEV(accessors, D, T);
  TEST_HOSTDEV(boundingBox, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox, T);
    TEST(aabb_constructor<T>);
    TEST(id_array_constructor<T>);
  }
}

auto
main() -> int
{
  RUN_SUITE((RectilinearGrid<1, float>));
  RUN_SUITE((RectilinearGrid<2, float>));
  RUN_SUITE((RectilinearGrid<3, float>));

  RUN_SUITE((RectilinearGrid<1, double>));
  RUN_SUITE((RectilinearGrid<2, double>));
  RUN_SUITE((RectilinearGrid<3, double>));
  return 0;
}
