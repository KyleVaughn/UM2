#include <um2/mesh/RectilinearPartition.hpp>

#include "../test_macros.hpp"

template <Size D, typename T, std::integral P>
HOSTDEV static constexpr auto
makePartition() -> um2::RectilinearPartition<D, T, P>
{
  um2::RectilinearPartition<D, T, P> partition;
  if constexpr (D >= 1) {
    partition.grid.divs[0] = {0, 1};
  }
  if constexpr (D >= 2) {
    partition.grid.divs[1] = {0, 1, 2};
  }
  if constexpr (D >= 3) {
    partition.grid.divs[2] = {0, 1, 2, 3};
  }
  if constexpr (D == 1) {
    partition.children = {1};
  } else if constexpr (D == 2) {
    partition.children = {1, 2};
  } else if constexpr (D == 3) {
    partition.children = {1, 2, 3, 4, 5, 6};
  } else {
    static_assert(!D, "Invalid dimension");
  }
  return partition;
}

template <Size D, typename T, std::integral P>
HOSTDEV
TEST_CASE(clear)
{
  um2::RectilinearPartition<D, T, P> part = makePartition<D, T, P>();
  part.clear();
  for (Size i = 0; i < D; ++i) {
    ASSERT(part.grid.divs[i].empty());
  }
  ASSERT(part.children.empty());
}

template <typename T, std::integral P>
TEST_CASE(id_array_constructor)
{
  std::vector<std::vector<Size>> const ids = {
      {0, 1, 2, 0},
      {0, 2, 0, 2},
      {0, 1, 0, 1},
  };
  std::vector<um2::Vec2<T>> const dxdy = {
      {2, 1},
      {2, 1},
      {2, 1},
      {2, 1},
  };
  um2::Vector<P> const expected = {0, 1, 0, 1, 0, 2, 0, 2, 0, 1, 2, 0};
  um2::RectilinearPartition2<T, P> const part(dxdy, ids);

  ASSERT(part.grid.divs[0].size() == 5);
  T const xref[5] = {0, 2, 4, 6, 8};
  for (Size i = 0; i < 5; ++i) {
    ASSERT_NEAR(part.grid.divs[0][i], xref[i], static_cast<T>(1e-6));
  }
  ASSERT(part.grid.divs[1].size() == 4);
  T const yref[4] = {0, 1, 2, 3};
  for (Size i = 0; i < 4; ++i) {
    ASSERT_NEAR(part.grid.divs[1][i], yref[i], static_cast<T>(1e-6));
  }
  for (Size i = 0; i < 12; ++i) {
    ASSERT(part.children[i] == expected[i]);
  }
}

template <typename T, std::integral P>
HOSTDEV
TEST_CASE(getBoxAndChild)
{
  // Declare some variables to avoid a bunch of static casts.
  T const three = static_cast<T>(3);
  T const two = static_cast<T>(2);
  T const one = static_cast<T>(1);
  T const half = static_cast<T>(0.5);
  T const forth = static_cast<T>(0.25);
  um2::RectilinearPartition2<T, P> part;
  part.grid.divs[0] = {1.0, 1.5, 2.0, 2.5, 3.0};
  part.grid.divs[1] = {-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
  part.children.resize(32);
  for (Size i = 0; i < 32; ++i) {
    part.children[i] = static_cast<P>(i);
  }
  um2::AxisAlignedBox2<T> box = part.getBox(0, 0);
  um2::AxisAlignedBox2<T> box_ref = {
      {         1,             -1},
      {one + half, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.getBox(1, 0);
  //{ { 1.5, -1.0 }, { 2.0, -0.75 } };
  box_ref = {
      {one + half,           -one},
      {       two, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.getBox(3, 0);
  // box_ref = { { 2.5, -1.0 }, { 3.0, -0.75 } };
  box_ref = {
      {two + half,           -one},
      {     three, -three * forth}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.getBox(0, 1);
  // box_ref = { { 1.0, -0.75 }, { 1.5, -0.5 } };
  box_ref = {
      {       one, -three * forth},
      {one + half,          -half}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.getBox(0, 7);
  // box_ref = { { 1.0, 0.75 }, { 1.5, 1.0 } };
  box_ref = {
      {       one, three * forth},
      {one + half,           one}
  };
  ASSERT(isApprox(box, box_ref));
  box = part.getBox(3, 7);
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
template <Size D, typename T, std::integral P>
MAKE_CUDA_KERNEL(clear, D, T, P)

template <typename T, std::integral P>
MAKE_CUDA_KERNEL(getBoxAndChild, T, P)
#endif

template <Size D, typename T, std::integral P>
TEST_SUITE(RectilinearPartition)
{
  TEST_HOSTDEV(clear, 1, 1, D, T, P);
  TEST_HOSTDEV(getBoxAndChild, 1, 1, T, P);
  if constexpr (D == 2) {
    TEST((id_array_constructor<T, P>));
  }
}

auto
main() -> int
{
  RUN_SUITE((RectilinearPartition<1, float, uint16_t>));
  RUN_SUITE((RectilinearPartition<1, double, uint16_t>));
  RUN_SUITE((RectilinearPartition<1, float, uint32_t>));
  RUN_SUITE((RectilinearPartition<1, double, uint32_t>));
  RUN_SUITE((RectilinearPartition<1, float, uint64_t>));
  RUN_SUITE((RectilinearPartition<1, double, uint64_t>));

  RUN_SUITE((RectilinearPartition<2, float, uint16_t>));
  RUN_SUITE((RectilinearPartition<2, double, uint16_t>));
  RUN_SUITE((RectilinearPartition<2, float, uint32_t>));
  RUN_SUITE((RectilinearPartition<2, double, uint32_t>));
  RUN_SUITE((RectilinearPartition<2, float, uint64_t>));
  RUN_SUITE((RectilinearPartition<2, double, uint64_t>));
  return 0;
}
