#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/pair.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(test_pair)
{
  um2::Pair<int, int> p{1, 2};
  ASSERT(p.first == 1);
  ASSERT(p.second == 2);

  um2::Pair<int, int> const p2{p};
  ASSERT(p2.first == 1);
  ASSERT(p2.second == 2);

  um2::Pair<int, int> const p3{um2::move(p)};
  ASSERT(p3.first == 1);
  ASSERT(p3.second == 2);

  um2::Pair<int, int> p4(1, 2); 
  ASSERT(p4.first == 1);
  ASSERT(p4.second == 2);

  ASSERT(p2 == p3);
  p4.first = 3;
  ASSERT(p2 != p4);
  ASSERT(p2 < p4);
  ASSERT(p4 > p2);
  p4.first = 1;
  p4.second = 3;
  ASSERT(p2 < p4);
  ASSERT(p4 > p2);
}
MAKE_CUDA_KERNEL(test_pair);

TEST_SUITE(pair) { TEST_HOSTDEV(test_pair); }

auto
main() -> int
{
  RUN_SUITE(pair);
  return 0;
}
