#include <um2/mesh/RegularGrid.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeGrid() -> um2::RegularGrid<D, T>
{
  static_assert(1 <= D && D <= 3, "D must be in [1, 3]");
  um2::RegularGrid<D, T> grid;
  if constexpr (D >= 1) {
    grid.minima[0] = 1;
    grid.spacing[0] = 1;
    grid.num_cells[0] = 1;
  }
  if constexpr (D >= 2) {
    grid.minima[1] = 2;
    grid.spacing[1] = 2;
    grid.num_cells[1] = 2;
  }
  if constexpr (D >= 3) {
    grid.minima[2] = 3;
    grid.spacing[2] = 3;
    grid.num_cells[2] = 3;
  }
  return grid;
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
//
// template <Size D, typename T>
// HOSTDEV TEST_CASE(bounding_box)
//{
//   um2::RegularGrid<D, T> const grid = makeGrid<D, T>();
//   um2::AxisAlignedBox<D, T> box = boundingBox(grid);
//   if constexpr (D >= 1) {
//     ASSERT_NEAR(box.minima[0], xMin(grid), static_cast<T>(1e-6));
//     ASSERT_NEAR(box.maxima[0], xMax(grid), static_cast<T>(1e-6));
//   }
//   if constexpr (D >= 2) {
//     ASSERT_NEAR(box.minima[1], yMin(grid), static_cast<T>(1e-6));
//     ASSERT_NEAR(box.maxima[1], yMax(grid), static_cast<T>(1e-6));
//   }
//   if constexpr (D >= 3) {
//     ASSERT_NEAR(box.minima[2], zMin(grid), static_cast<T>(1e-6));
//     ASSERT_NEAR(box.maxima[2], zMax(grid), static_cast<T>(1e-6));
//   }
// }
//
// template <typename T>
// HOSTDEV TEST_CASE(getBox)
//{
//   // Declare some variables to avoid a bunch of static casts.
//   T const three = static_cast<T>(3);
//   T const two = static_cast<T>(2);
//   T const one = static_cast<T>(1);
//   T const half = static_cast<T>(0.5);
//   T const forth = static_cast<T>(0.25);
//   um2::Point2<T> minima = {1, -1};
//   um2::Vec2<T> spacing = {half, forth};
//   um2::Vec2<Size> num_cells = {4, 8};
//   um2::RegularGrid2<T> grid(minima, spacing, num_cells);
//   um2::AxisAlignedBox2<T> box = grid.getBox(0, 0);
//   um2::AxisAlignedBox2<T> box_ref = {
//       {         1,             -1},
//       {one + half, -three * forth}
//   };
//   ASSERT_TRUE(isApprox(box, box_ref));
//   box = grid.getBox(1, 0);
//   //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
//   box_ref = {
//       {one + half,           -one},
//       {       two, -three * forth}
//   };
//   ASSERT_TRUE(isApprox(box, box_ref));
//   box = grid.getBox(3, 0);
//   // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
//   box_ref = {
//       {two + half,           -one},
//       {     three, -three * forth}
//   };
//   ASSERT_TRUE(isApprox(box, box_ref));
//   box = grid.getBox(0, 1);
//   // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
//   box_ref = {
//       {       one, -three * forth},
//       {one + half,          -half}
//   };
//   ASSERT_TRUE(isApprox(box, box_ref));
//   box = grid.getBox(0, 7);
//   // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
//   box_ref = {
//       {       one, three * forth},
//       {one + half,           one}
//   };
//   ASSERT_TRUE(isApprox(box, box_ref));
//   box = grid.getBox(3, 7);
//   // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
//   box_ref = {
//       {two + half, three * forth},
//       {     three,           one}
//   };
//   ASSERT_TRUE(isApprox(box, box_ref));
// }

#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(constructor, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(accessors, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T)

template <Size D, typename T>
MAKE_CUDA_KERNEL(getBox, D, T)

#endif

template <Size D, typename T>
TEST_SUITE(RegularGrid)
{
  TEST_HOSTDEV(constructor, 1, 1, D, T);
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  //  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  //  if constexpr (D == 2) {
  //    TEST_HOSTDEV((getBox<T>));
  //  }
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
