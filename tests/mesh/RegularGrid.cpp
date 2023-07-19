#include <um2/mesh/RegularGrid.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeGrid() -> um2::RegularGrid<D, T>
{
  static_assert(1 <= D && D <= 3, "D must be in [1, 3]");
  um2::Point<D, T> minima;
  um2::Point<D, T> spacing;
  um2::Point<D, Size> num_cells;
  for (Size i = 0; i < D; ++i) {
    minima[i] = static_cast<T>(i + 1);
    spacing[i] = static_cast<T>(i + 1);
    num_cells[i] = i + 1;
  }
  return {minima, spacing, num_cells};
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(constructor)
{
  um2::RegularGrid<D, T> grid_ref = makeGrid<D, T>();
  um2::RegularGrid<D, T> grid(grid_ref.minima, grid_ref.spacing, grid_ref.num_cells);
  if constexpr (D >= 1) {
    ASSERT_NEAR(grid.minima[0], grid_ref.minima[0], static_cast<T>(1e-6));
    ASSERT_NEAR(grid.spacing[0], grid_ref.spacing[0], static_cast<T>(1e-6));
    ASSERT(grid.num_cells[0] == grid_ref.num_cells[0]);
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(grid.minima[1], grid_ref.minima[1], static_cast<T>(1e-6));
    ASSERT_NEAR(grid.spacing[1], grid_ref.spacing[1], static_cast<T>(1e-6));
    ASSERT(grid.num_cells[1] == grid_ref.num_cells[1]);
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(grid.minima[2], grid_ref.minima[2], static_cast<T>(1e-6));
    ASSERT_NEAR(grid.spacing[2], grid_ref.spacing[2], static_cast<T>(1e-6));
    ASSERT(grid.num_cells[2] == grid_ref.num_cells[2]);
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RegularGrid<D, T> grid = makeGrid<D, T>();
  if constexpr (D >= 1) {
    T const xmin = grid.minima[0];
    ASSERT_NEAR(grid.xMin(), xmin, static_cast<T>(1e-6));
    T const dx = grid.spacing[0];
    ASSERT_NEAR(grid.dx(), dx, static_cast<T>(1e-6));
    auto const nx = grid.num_cells[0];
    ASSERT(grid.numXCells() == nx);
    ASSERT_NEAR(grid.width(), dx * static_cast<T>(nx), static_cast<T>(1e-6));
    ASSERT_NEAR(grid.xMax(), xmin + dx * static_cast<T>(nx), static_cast<T>(1e-6));
  }
  if constexpr (D >= 2) {
    T const ymin = grid.minima[1];
    ASSERT_NEAR(grid.yMin(), ymin, static_cast<T>(1e-6));
    T const dy = grid.spacing[1];
    ASSERT_NEAR(grid.dy(), dy, static_cast<T>(1e-6));
    auto const ny = grid.num_cells[1];
    ASSERT(grid.numYCells() == ny);
    ASSERT_NEAR(grid.height(), dy * static_cast<T>(ny), static_cast<T>(1e-6));
    ASSERT_NEAR(grid.yMax(), ymin + dy * static_cast<T>(ny), static_cast<T>(1e-6));
  }
  if constexpr (D >= 3) {
    T const zmin = grid.minima[2];
    ASSERT_NEAR(grid.zMin(), zmin, static_cast<T>(1e-6));
    T const dz = grid.spacing[2];
    ASSERT_NEAR(grid.dz(), dz, static_cast<T>(1e-6));
    auto const nz = grid.num_cells[2];
    ASSERT(grid.numZCells() == nz);
    ASSERT_NEAR(grid.depth(), dz * static_cast<T>(nz), static_cast<T>(1e-6));
    ASSERT_NEAR(grid.zMax(), zmin + dz * static_cast<T>(nz), static_cast<T>(1e-6));
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RegularGrid<D, T> const grid = makeGrid<D, T>();
  um2::AxisAlignedBox<D, T> box = grid.boundingBox();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.minima[0], grid.xMin(), static_cast<T>(1e-6));
    ASSERT_NEAR(box.maxima[0], grid.xMax(), static_cast<T>(1e-6));
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.minima[1], grid.yMin(), static_cast<T>(1e-6));
    ASSERT_NEAR(box.maxima[1], grid.yMax(), static_cast<T>(1e-6));
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.minima[2], grid.zMin(), static_cast<T>(1e-6));
    ASSERT_NEAR(box.maxima[2], grid.zMax(), static_cast<T>(1e-6));
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
  T const ahalf = static_cast<T>(0.5);
  T const forth = static_cast<T>(0.25);
  um2::Point2<T> minima = {1, -1};
  um2::Vec2<T> spacing = {ahalf, forth};
  um2::Vec2<Size> num_cells = {4, 8};
  um2::RegularGrid2<T> grid(minima, spacing, num_cells);
  um2::AxisAlignedBox2<T> box = grid.getBox(0, 0);
  um2::AxisAlignedBox2<T> box_ref = {
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

template <typename T>
HOSTDEV
TEST_CASE(getRangeContaining)
{
  um2::Point2<T> minima(1, -1);
  um2::Vec2<T> spacing(2, 1);
  um2::Vec2<Size> num_cells(5, 8);
  // Grid ranges from 1 to 11 in x and -1 to 7 in y.
  um2::RegularGrid2<T> grid(minima, spacing, num_cells);

  // A box in a single cell.
  um2::AxisAlignedBox2<T> box0({static_cast<T>(3.1), static_cast<T>(1.1)},
                               {static_cast<T>(3.9), static_cast<T>(1.9)});
  um2::Vec<4, Size> range0 = grid.getRangeContaining(box0);
  ASSERT(range0[0] == 1);
  ASSERT(range0[1] == 2);
  ASSERT(range0[2] == 1);
  ASSERT(range0[3] == 2);

  // A box with perfect alignment.
  um2::AxisAlignedBox2<T> box1({static_cast<T>(3), static_cast<T>(1)},
                               {static_cast<T>(5), static_cast<T>(2)});
  um2::Vec<4, Size> range1 = grid.getRangeContaining(box1);
  ASSERT(range1[0] == 0 || range1[0] == 1);
  ASSERT(range1[1] == 1 || range1[1] == 2);
  ASSERT(range1[2] == 1 || range1[2] == 2); // Valid in either cell.
  ASSERT(range1[3] == 2 || range1[3] == 3);

  // A box in multiple cells.
  um2::AxisAlignedBox2<T> box2({static_cast<T>(3.1), static_cast<T>(1.1)},
                               {static_cast<T>(5.9), static_cast<T>(1.9)});
  um2::Vec<4, Size> range2 = grid.getRangeContaining(box2);
  ASSERT(range2[0] == 1);
  ASSERT(range2[1] == 2);
  ASSERT(range2[2] == 2);
  ASSERT(range2[3] == 2);

  // A box in 4 cells.
  um2::AxisAlignedBox2<T> box3({static_cast<T>(3.1), static_cast<T>(1.1)},
                               {static_cast<T>(5.9), static_cast<T>(2.9)});
  um2::Vec<4, Size> range3 = grid.getRangeContaining(box3);
  ASSERT(range3[0] == 1);
  ASSERT(range3[1] == 2);
  ASSERT(range3[2] == 2);
  ASSERT(range3[3] == 3);
}

#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(constructor, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(accessors, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T)

template <typename T>
MAKE_CUDA_KERNEL(getBox, T)

template <typename T>
MAKE_CUDA_KERNEL(getRangeContaining, T)

#endif

template <Size D, typename T>
TEST_SUITE(RegularGrid)
{
  TEST_HOSTDEV(constructor, 1, 1, D, T);
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox, 1, 1, T);
    TEST_HOSTDEV(getRangeContaining, 1, 1, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((RegularGrid<1, float>));
  RUN_SUITE((RegularGrid<1, double>));
  RUN_SUITE((RegularGrid<2, float>));
  RUN_SUITE((RegularGrid<2, double>));
  RUN_SUITE((RegularGrid<3, float>));
  RUN_SUITE((RegularGrid<3, double>));
  return 0;
}
