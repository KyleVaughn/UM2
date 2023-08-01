#include <um2/mesh/RectilinearGrid.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeGrid() -> um2::RectilinearGrid<D, T>
{
  um2::RectilinearGrid<D, T> grid;
  if constexpr (D >= 1) {
    grid.divs[0] = {0, 1};
  }
  if constexpr (D >= 2) {
    grid.divs[1] = {0, 1, 2};
  }
  if constexpr (D >= 3) {
    grid.divs[2] = {0, 1, 2, 3};
  }
  return grid;
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(clear)
{
  um2::RectilinearGrid<D, T> grid = makeGrid<D, T>();
  grid.clear();
  for (Size i = 0; i < D; ++i) {
    ASSERT(grid.divs[i].empty());
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RectilinearGrid<D, T> grid = makeGrid<D, T>();
  um2::Vec<D, Size> const ncells = grid.numCells();
  if constexpr (D >= 1) {
    auto const nx = 1;
    ASSERT_NEAR(grid.xMin(), grid.divs[0][0], static_cast<T>(1e-6));
    ASSERT_NEAR(grid.xMax(), grid.divs[0][nx], static_cast<T>(1e-6));
    ASSERT(grid.numXCells() == nx);
    ASSERT(ncells[0] == nx);
    ASSERT_NEAR(grid.width(), grid.divs[0][nx] - grid.divs[0][0], static_cast<T>(1e-6));
  }
  if constexpr (D >= 2) {
    auto const ny = 2;
    ASSERT_NEAR(grid.yMin(), grid.divs[1][0], static_cast<T>(1e-6));
    ASSERT_NEAR(grid.yMax(), grid.divs[1][ny], static_cast<T>(1e-6));
    ASSERT(grid.numYCells() == ny);
    ASSERT(ncells[1] == ny);
    ASSERT_NEAR(grid.height(), grid.divs[1][ny] - grid.divs[1][0], static_cast<T>(1e-6));
  }
  if constexpr (D >= 3) {
    auto const nz = 3;
    ASSERT_NEAR(grid.zMin(), grid.divs[2][0], static_cast<T>(1e-6));
    ASSERT_NEAR(grid.zMax(), grid.divs[2][nz], static_cast<T>(1e-6));
    ASSERT(grid.numZCells() == nz);
    ASSERT(ncells[2] == nz);
    ASSERT_NEAR(grid.depth(), grid.divs[2][nz] - grid.divs[2][0], static_cast<T>(1e-6));
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RectilinearGrid<D, T> grid = makeGrid<D, T>();
  um2::AxisAlignedBox<D, T> box = grid.boundingBox();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.minima[0], grid.divs[0][0], static_cast<T>(1e-6));
    ASSERT_NEAR(box.maxima[0], grid.divs[0][1], static_cast<T>(1e-6));
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.minima[1], grid.divs[1][0], static_cast<T>(1e-6));
    ASSERT_NEAR(box.maxima[1], grid.divs[1][2], static_cast<T>(1e-6));
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.minima[2], grid.divs[2][0], static_cast<T>(1e-6));
    ASSERT_NEAR(box.maxima[2], grid.divs[2][3], static_cast<T>(1e-6));
  }
}

template <typename T>
HOSTDEV
TEST_CASE(getBox)
{
  // Declare some variables to avoid a bunch of static casts.
  T const three = static_cast<T>(3);
  T const two = static_cast<T>(2);
  T const one = static_cast<T>(1);
  T const half = static_cast<T>(0.5);
  T const forth = static_cast<T>(0.25);
  um2::RectilinearGrid2<T> grid;
  grid.divs[0] = {1.0, 1.5, 2.0, 2.5, 3.0};
  grid.divs[1] = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
  um2::AxisAlignedBox2<T> box = grid.getBox(0, 0);
  um2::AxisAlignedBox2<T> box_ref = {
      {         1,             -1},
      {one + half, -three * forth}
  };
  ASSERT(um2::isApprox(box, box_ref));
  box = grid.getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + half,           -one},
      {       two, -three * forth}
  };
  ASSERT(um2::isApprox(box, box_ref));
  box = grid.getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + half,           -one},
      {     three, -three * forth}
  };
  ASSERT(um2::isApprox(box, box_ref));
  box = grid.getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {       one, -three * forth},
      {one + half,          -half}
  };
  ASSERT(um2::isApprox(box, box_ref));
  box = grid.getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {       one, three * forth},
      {one + half,           one}
  };
  ASSERT(um2::isApprox(box, box_ref));
  box = grid.getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + half, three * forth},
      {     three,           one}
  };
  ASSERT(um2::isApprox(box, box_ref));
}

template <typename T>
TEST_CASE(aabb2_constructor)
{
    um2::AxisAlignedBox2<T> const b00(um2::Point2<T>(0, 0), um2::Point2<T>(1, 1));
    um2::AxisAlignedBox2<T> const b10(um2::Point2<T>(1, 0), um2::Point2<T>(2, 1));
    um2::AxisAlignedBox2<T> const b01(um2::Point2<T>(0, 1), um2::Point2<T>(1, 2));
    um2::AxisAlignedBox2<T> const b11(um2::Point2<T>(1, 1), um2::Point2<T>(2, 2));
    um2::AxisAlignedBox2<T> const b02(um2::Point2<T>(0, 2), um2::Point2<T>(1, 3));
    um2::AxisAlignedBox2<T> const b12(um2::Point2<T>(1, 2), um2::Point2<T>(2, 3));
    um2::Vector<um2::AxisAlignedBox2<T>> const boxes = { b00, b10, b01, b11, b02, b12 };
    um2::RectilinearGrid2<T> grid(boxes);

    ASSERT(grid.divs[0].size() == 3);
    T const xref[3] = { 0, 1, 2 };
    for (Size i = 0; i < 3; ++i) {
        ASSERT_NEAR(grid.divs[0][i], xref[i], static_cast<T>(1e-6));
    }

    ASSERT(grid.divs[1].size() == 4);
    T const yref[4] = { 0, 1, 2, 3 };
    for (Size i = 0; i < 4; ++i) {
        ASSERT_NEAR(grid.divs[1][i], yref[i], static_cast<T>(1e-6));
    }

    um2::RectilinearGrid2<T> grid2(b01);
    ASSERT(grid2.divs[0].size() == 2);
    ASSERT(grid2.divs[1].size() == 2);
    ASSERT_NEAR(grid2.divs[0][0], 0, static_cast<T>(1e-6));
    ASSERT_NEAR(grid2.divs[0][1], 1, static_cast<T>(1e-6));
    ASSERT_NEAR(grid2.divs[1][0], 1, static_cast<T>(1e-6));
    ASSERT_NEAR(grid2.divs[1][1], 2, static_cast<T>(1e-6));
}

template <typename T>
TEST_CASE(id_array_constructor)
{
    std::vector<std::vector<Size>> const ids = {
        { 0, 1, 2, 0 },
        { 0, 2, 0, 2 },
        { 0, 1, 0, 1 },
    };
    std::vector<um2::Vec2<T>> const dxdy = {
        { 2, 1 },
        { 2, 1 },
        { 2, 1 },
        { 2, 1 },
    };
    um2::RectilinearGrid2<T> grid(dxdy, ids);

    ASSERT(grid.divs[0].size() == 5);
    T const xref[5] = { 0, 2, 4, 6, 8 };
    for (Size i = 0; i < 5; ++i) {
        ASSERT_NEAR(grid.divs[0][i], xref[i], static_cast<T>(1e-6));
    }
    ASSERT(grid.divs[1].size() == 4);
    T const yref[4] = { 0, 1, 2, 3 };
    for (Size i = 0; i < 4; ++i) {
        ASSERT_NEAR(grid.divs[1][i], yref[i], static_cast<T>(1e-6));
    }
  }
#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(clear, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(accessors, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T)

template <typename T>
MAKE_CUDA_KERNEL(getBox, T)
#endif

template <Size D, typename T>
TEST_SUITE(RectilinearGrid)
{
  TEST_HOSTDEV(clear, 1, 1, D, T);
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox, 1, 1, T);
    TEST((aabb2_constructor<T>)  );
    TEST((id_array_constructor<T>)  );
  }
}

auto
main() -> int
{
  RUN_SUITE((RectilinearGrid<1, float>));
  RUN_SUITE((RectilinearGrid<1, double>));
  RUN_SUITE((RectilinearGrid<2, float>));
  RUN_SUITE((RectilinearGrid<2, double>));
  RUN_SUITE((RectilinearGrid<3, float>));
  RUN_SUITE((RectilinearGrid<3, double>));
  return 0;
}
