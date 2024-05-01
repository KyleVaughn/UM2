#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/pair.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(test_pair)
{
  um2::Pair<int, int> p{1, 2};

  // Copy constructor
  um2::Pair<int, int> const p2{p};
  ASSERT(p2.first == 1);
  ASSERT(p2.second == 2);

  // Move constructor
  um2::Pair<int, int> const p3{um2::move(p)};
  ASSERT(p3.first == 1);
  ASSERT(p3.second == 2);

  // Explicit constructor
  um2::Pair<int, int> p4(1, 2);
  ASSERT(p4.first == 1);
  ASSERT(p4.second == 2);

  // Copy assignment
  um2::Pair<int, int> p5 = p4;
  ASSERT(p5.first == 1);
  ASSERT(p5.second == 2);

  // Move assignment
  um2::Pair<int, int> const p6 = um2::move(p5);
  ASSERT(p6.first == 1);
  ASSERT(p6.second == 2);

  // Relational operators
  ASSERT(p2 == p3);
  ASSERT(p2 >= p3);
  ASSERT(p2 <= p3);
  p4.first = 3;
  ASSERT(p2 != p4);
  ASSERT(p2 < p4);
  ASSERT(p2 <= p4);
  ASSERT(p4 > p2);
  ASSERT(p4 >= p2);
  p4.first = 1;
  p4.second = 3;
  ASSERT(p2 < p4);
  ASSERT(p4 > p2);
}
MAKE_CUDA_KERNEL(test_pair);

HOSTDEV
TEST_CASE(test_pair_constexpr)
{
  um2::Pair<int, int> constexpr p{1, 2};

  // Copy constructor
  um2::Pair<int, int> constexpr p2{p};
  static_assert(p2.first == 1);
  static_assert(p2.second == 2);

  // Explicit constructor
  um2::Pair<int, int> constexpr p3(1, 3);
  static_assert(p3.first == 1);
  static_assert(p3.second == 3);

  // Copy assignment
  um2::Pair<int, int> constexpr p4 = p3;
  static_assert(p4.first == 1);
  static_assert(p4.second == 3);

  // Relational operators
  static_assert(p == p2);
  static_assert(p2 <= p3);
  static_assert(p3 >= p2);
  static_assert(p2 != p4);
  static_assert(p2 < p4);
  static_assert(p4 > p2);
}
MAKE_CUDA_KERNEL(test_pair_constexpr)

TEST_SUITE(pair)
{
  TEST_HOSTDEV(test_pair);
  TEST_HOSTDEV(test_pair_constexpr);
}

auto
main() -> int
{
  RUN_SUITE(pair);
  return 0;
}
