#include <um2/mesh/rectilinear_partition.hpp>

#include "../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

template <I D, std::integral P>
HOSTDEV constexpr auto
makePartition() -> um2::RectilinearPartition<D, P>
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

  um2::Vector<P> children;
  if constexpr (D == 1) {
    children = {1};
  } else if constexpr (D == 2) {
    children = {1, 2};
  } else if constexpr (D == 3) {
    children = {1, 2, 3, 4, 5, 6};
  } else {
    static_assert(!D, "Invalid dimension");
  }

  um2::RectilinearPartition<D, P> partition(grid, children);
  return partition;
}

template <I D, std::integral P>
HOSTDEV
TEST_CASE(clear)
{
  um2::RectilinearPartition<D, P> part = makePartition<D, P>();
  part.clear();
  for (I i = 0; i < D; ++i) {
    ASSERT(part.grid().divs(i).empty());
  }
  ASSERT(part.children().empty());
}

template <std::integral P>
TEST_CASE(id_array_constructor)
{
  um2::Vector<um2::Vector<I>> const ids = {
      {0, 1, 2, 0},
      {0, 2, 0, 2},
      {0, 1, 0, 1},
  };
  um2::Vector<um2::Vec2<F>> const dxdy = {
      {2, 1},
      {2, 1},
      {2, 1},
      {2, 1},
  };
  um2::Vector<P> const expected = {0, 1, 0, 1, 0, 2, 0, 2, 0, 1, 2, 0};
  um2::RectilinearPartition2<P> const part(dxdy, ids);

  ASSERT(part.grid().divs(0).size() == 5);
  F const xref[5] = {0, 2, 4, 6, 8};
  for (I i = 0; i < 5; ++i) {
    ASSERT_NEAR(part.grid().divs(0)[i], xref[i], eps);
  }
  ASSERT(part.grid().divs(1).size() == 4);
  F const yref[4] = {0, 1, 2, 3};
  for (I i = 0; i < 4; ++i) {
    ASSERT_NEAR(part.grid().divs(1)[i], yref[i], eps);
  }
  for (I i = 0; i < 12; ++i) {
    ASSERT(part.children()[i] == expected[i]);
  }
}

template <std::integral P>
HOSTDEV
TEST_CASE(getBoxAndChild)
{
  // Declare some variables to avoid a bunch of static casts.
  F const three = static_cast<F>(3);
  F const two = static_cast<F>(2);
  F const one = static_cast<F>(1);
  F const half = static_cast<F>(1) / static_cast<F>(2);
  F const forth = static_cast<F>(1) / static_cast<F>(4);
  um2::RectilinearGrid2 grid;
  grid.divs(0) = {1.0, 1.5, 2.0, 2.5, 3.0};
  grid.divs(1) = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
  um2::Vector<P> children(32);
  for (I i = 0; i < 32; ++i) {
    children[i] = static_cast<P>(i);
  }

  um2::RectilinearPartition2<P> part(grid, children);
  um2::AxisAlignedBox2 box = part.grid().getBox(0, 0);
  um2::AxisAlignedBox2 box_ref = {
      {         1,             -1},
      {one + half, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + half,           -one},
      {       two, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + half,           -one},
      {     three, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {       one, -three * forth},
      {one + half,          -half}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {       one, three * forth},
      {one + half,           one}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.grid().getBox(3, 7);
  // box_ref = { { 2.5, 0.75 }, { 3.0, 1.0 } };
  box_ref = {
      {two + half, three * forth},
      {     three,           one}
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

template <I D, std::integral P>
MAKE_CUDA_KERNEL(clear, D, P)

template <typename T, std::integral P>
MAKE_CUDA_KERNEL(getBoxAndChild, P)

#endif

template <I D, std::integral P>
TEST_SUITE(RectilinearPartition)
{
  TEST_HOSTDEV(clear, 1, 1, D, P);
  TEST_HOSTDEV(getBoxAndChild, 1, 1, P);
  if constexpr (D == 2) {
    TEST(id_array_constructor<P>);
  }
}

auto
main() -> int
{
  RUN_SUITE((RectilinearPartition<1, uint16_t>));
  RUN_SUITE((RectilinearPartition<1, uint32_t>));
  RUN_SUITE((RectilinearPartition<1, uint64_t>));

  RUN_SUITE((RectilinearPartition<2, uint16_t>));
  RUN_SUITE((RectilinearPartition<2, uint32_t>));
  RUN_SUITE((RectilinearPartition<2, uint64_t>));
  return 0;
}
