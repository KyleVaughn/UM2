#include <um2/mesh/regular_partition.hpp>

#include "../test_macros.hpp"

auto constexpr eps = castIfNot<Float>(1e-6);

template <Int D, typename P>
HOSTDEV constexpr auto
makePart() -> um2::RegularPartition<D, P>
{
  static_assert(1 <= D && D <= 3, "D must be in [1, 3]");
  um2::Point<D> minima;
  um2::Point<D> spacing;
  um2::Vec<D, Int> num_cells;
  for (Int i = 0; i < D; ++i) {
    minima[i] = static_cast<Float>(i + 1);
    spacing[i] = static_cast<Float>(i + 1);
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
  um2::RegularGrid<D> const grid(minima, spacing, num_cells);
  um2::RegularPartition<D, P> part(grid, um2::move(children));
  return part;
}

template <Int D, typename P>
HOSTDEV
TEST_CASE(accessors)
{
  um2::RegularPartition<D, P> const part = makePart<D, P>();
  if constexpr (D >= 1) {
    auto const xmin = static_cast<Float>(1);
    ASSERT_NEAR(part.grid().xMin(), xmin, eps);
    auto const dx = static_cast<Float>(1);
    ASSERT_NEAR(part.grid().dx(), dx, eps);
    Int const nx = 1;
    ASSERT(part.grid().numXCells() == nx);
    ASSERT_NEAR(part.grid().width(), dx * static_cast<Float>(nx), eps);
    ASSERT_NEAR(part.grid().xMax(), xmin + dx * static_cast<Float>(nx), eps);
  }
  if constexpr (D >= 2) {
    auto const ymin = static_cast<Float>(2);
    ASSERT_NEAR(part.grid().yMin(), ymin, eps);
    auto const dy = static_cast<Float>(2);
    ASSERT_NEAR(part.grid().dy(), dy, eps);
    Int const ny = 2;
    ASSERT(part.grid().numYCells() == ny);
    ASSERT_NEAR(part.grid().height(), dy * static_cast<Float>(ny), eps);
    ASSERT_NEAR(part.grid().yMax(), ymin + dy * static_cast<Float>(ny), eps);
  }
  if constexpr (D >= 3) {
    auto const zmin = static_cast<Float>(3);
    ASSERT_NEAR(part.grid().zMin(), zmin, eps);
    auto const dz = static_cast<Float>(3);
    ASSERT_NEAR(part.grid().dz(), dz, eps);
    Int const nz = 3;
    ASSERT(part.grid().numZCells() == nz);
    ASSERT_NEAR(part.grid().depth(), dz * static_cast<Float>(nz), eps);
    ASSERT_NEAR(part.grid().zMax(), zmin + dz * static_cast<Float>(nz), eps);
  }
}

template <Int D, typename P>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::RegularPartition<D, P> const part = makePart<D, P>();
  um2::AxisAlignedBox<D> const box = part.grid().boundingBox();
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

template <typename P>
HOSTDEV
TEST_CASE(getBox_and_getChild)
{
  // Declare some variables to avoid a bunch of static casts.
  auto const three = static_cast<Float>(3);
  auto const two = static_cast<Float>(2);
  auto const one = static_cast<Float>(1);
  auto const ahalf = static_cast<Float>(1) / static_cast<Float>(2);
  auto const forth = static_cast<Float>(1) / static_cast<Float>(4);
  um2::Point2 const minima = {1, -1};
  um2::Vec2<Float> const spacing = {ahalf, forth};
  um2::Vec2<Int> const num_cells = {4, 8};
  um2::RegularGrid2 const grid(minima, spacing, num_cells);
  um2::Vector<P> children(32);
  for (Int i = 0; i < 32; ++i) {
    children[i] = static_cast<P>(i);
  }
  um2::RegularPartition<2, P> part(grid, children);
  um2::AxisAlignedBox2 box = part.grid().getBox(0, 0);
  um2::AxisAlignedBox2 box_ref = {
      {          1,             -1},
      {one + ahalf, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = part.grid().getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + ahalf,           -one},
      {        two, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = part.grid().getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + ahalf,           -one},
      {      three, -three * forth}
  };
  ASSERT(box.isApprox(box_ref));
  box = part.grid().getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {        one, -three * forth},
      {one + ahalf,         -ahalf}
  };
  ASSERT(box.isApprox(box_ref));
  box = part.grid().getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {        one, three * forth},
      {one + ahalf,           one}
  };
  ASSERT(box.isApprox(box_ref));
  box = part.grid().getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + ahalf, three * forth},
      {      three,           one}
  };
  ASSERT(box.isApprox(box_ref));

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
template <Int D, typename P>
MAKE_CUDA_KERNEL(accessors, D, P)

template <Int D, typename P>
MAKE_CUDA_KERNEL(boundingBox, D, P)

template <typename P>
MAKE_CUDA_KERNEL(getBox_and_getChild, P)
#endif

template <Int D, typename P>
TEST_SUITE(RegularPartition)
{
  TEST_HOSTDEV(accessors, D, P);
  TEST_HOSTDEV(boundingBox, D, P);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getBox_and_getChild, P);
  }
}

auto
main() -> int
{
  RUN_SUITE((RegularPartition<1, int32_t>));
  RUN_SUITE((RegularPartition<2, int32_t>));
  RUN_SUITE((RegularPartition<3, int32_t>));
  RUN_SUITE((RegularPartition<1, uint64_t>));
  RUN_SUITE((RegularPartition<2, uint64_t>));
  RUN_SUITE((RegularPartition<3, uint64_t>));
  return 0;
}
