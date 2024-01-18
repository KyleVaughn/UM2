#include <um2/mesh/regular_partition.hpp>

#include "../test_macros.hpp"

template <Size D, typename T, typename P>
HOSTDEV constexpr auto
makePart() -> um2::RegularPartition<D, T, P>
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
  um2::Vector<P> children;
  if constexpr (D == 1) {
    children = {1};
  } else if constexpr (D == 2) {
    children = {1, 2};
  } else if constexpr (D == 3) {
    children = {1, 2, 3, 4, 5, 6};
  }
  um2::RegularGrid<D, T> const grid(minima, spacing, num_cells);
  um2::RegularPartition<D, T, P> part(grid, um2::move(children));
  return part;
}

template <Size D, typename T, typename P>
HOSTDEV
TEST_CASE(accessors)
{
  T constexpr eps = static_cast<T>(1e-6);
  um2::RegularPartition<D, T, P> const part = makePart<D, T, P>();
  if constexpr (D >= 1) {
    T const xmin = static_cast<T>(1);
    ASSERT_NEAR(part.grid().xMin(), xmin, eps);
    T const dx = static_cast<T>(1);
    ASSERT_NEAR(part.grid().dx(), dx, eps);
    Size const nx = 1;
    ASSERT(part.grid().numXCells() == nx);
    ASSERT_NEAR(part.grid().width(), dx * static_cast<T>(nx), eps);
    ASSERT_NEAR(part.grid().xMax(), xmin + dx * static_cast<T>(nx), eps);
  }
  if constexpr (D >= 2) {
    T const ymin = static_cast<T>(2);
    ASSERT_NEAR(part.grid().yMin(), ymin, eps);
    T const dy = static_cast<T>(2);
    ASSERT_NEAR(part.grid().dy(), dy, eps);
    Size const ny = 2;
    ASSERT(part.grid().numYCells() == ny);
    ASSERT_NEAR(part.grid().height(), dy * static_cast<T>(ny), eps);
    ASSERT_NEAR(part.grid().yMax(), ymin + dy * static_cast<T>(ny), eps);
  }
  if constexpr (D >= 3) {
    T const zmin = static_cast<T>(3);
    ASSERT_NEAR(part.grid().zMin(), zmin, eps);
    T const dz = static_cast<T>(3);
    ASSERT_NEAR(part.grid().dz(), dz, eps);
    Size const nz = 3;
    ASSERT(part.grid().numZCells() == nz);
    ASSERT_NEAR(part.grid().depth(), dz * static_cast<T>(nz), eps);
    ASSERT_NEAR(part.grid().zMax(), zmin + dz * static_cast<T>(nz), eps);
  }
}

template <Size D, typename T, typename P>
HOSTDEV
TEST_CASE(boundingBox)
{
  T constexpr eps = static_cast<T>(1e-6);
  um2::RegularPartition<D, T, P> const part = makePart<D, T, P>();
  um2::AxisAlignedBox<D, T> const box = part.grid().boundingBox();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.minima()[0], part.grid().xMin(), eps);
    ASSERT_NEAR(box.maxima()[0], part.grid().xMax(), eps);
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.minima()[1], part.grid().yMin(), eps);
    ASSERT_NEAR(box.maxima()[1], part.grid().yMax(), eps);
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.minima()[2], part.grid().zMin(), eps);
    ASSERT_NEAR(box.maxima()[2], part.grid().zMax(), eps);
  }
}

template <typename T, typename P>
HOSTDEV
TEST_CASE(getBox_and_getChild)
{
  // Declare some variables to avoid a bunch of static casts.
  T const three = static_cast<T>(3);
  T const two = static_cast<T>(2);
  T const one = static_cast<T>(1);
  T const ahalf = static_cast<T>(0.5);
  T const forth = static_cast<T>(0.25);
  um2::Point2<T> const minima = {1, -1};
  um2::Vec2<T> const spacing = {ahalf, forth};
  um2::Vec2<Size> const num_cells = {4, 8};
  um2::RegularGrid2<T> const grid(minima, spacing, num_cells);
  um2::Vector<P> children(32);
  for (Size i = 0; i < 32; ++i) {
    children[i] = static_cast<P>(i);
  }
  um2::RegularPartition<2, T, P> part(grid, children);
  um2::AxisAlignedBox2<T> box = part.grid().getBox(0, 0);
  um2::AxisAlignedBox2<T> box_ref = {
      {          1,             -1},
      {one + ahalf, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + ahalf,           -one},
      {        two, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + ahalf,           -one},
      {      three, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {        one, -three * forth},
      {one + ahalf,         -ahalf}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {        one, three * forth},
      {one + ahalf,           one}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + ahalf, three * forth},
      {      three,           one}
  };
  ASSERT(isApprox(box, box_ref));

  P child = part.getChild(0, 0);
  ASSERT(child == 0);
  child = part.getChild(1, 0);
  ASSERT(child == 1);
  child = part.getChild(3, 0);
  ASSERT(child == 3);
  child = part.getChild(0, 1);
  ASSERT(child == 4);
  child = part.getChild(0, 7);
  ASSERT(child == 28);
  child = part.getChild(3, 7);
  ASSERT(child == 31);
}

#if UM2_USE_CUDA
template <Size D, typename T, typename P>
MAKE_CUDA_KERNEL(accessors, D, T, P)

template <Size D, typename T, typename P>
MAKE_CUDA_KERNEL(boundingBox, D, T, P)

template <typename T, typename P>
MAKE_CUDA_KERNEL(getBox_and_getChild, T, P)
#endif

template <Size D, typename T, typename P>
TEST_SUITE(RegularPartition)
{
  TEST_HOSTDEV(accessors, 1, 1, D, T, P);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T, P);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox_and_getChild, 1, 1, T, P);
  }
}

auto
main() -> int
{
  RUN_SUITE((RegularPartition<1, float, int32_t>));
  RUN_SUITE((RegularPartition<2, float, int32_t>));
  RUN_SUITE((RegularPartition<3, float, int32_t>));
  RUN_SUITE((RegularPartition<1, double, uint64_t>));
  RUN_SUITE((RegularPartition<2, double, uint64_t>));
  RUN_SUITE((RegularPartition<3, double, uint64_t>));
  return 0;
}
