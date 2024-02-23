#include <um2/stdlib/algorithm/equal.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(test_equal)
{
  int a[] = {10, 20, 30};
  int b[] = {10, 20, 30};
  int c[] = {30, 20, 10};

  ASSERT(um2::equal(a, a + 3, b));
  ASSERT(!um2::equal(a, a + 3, c));
}

MAKE_CUDA_KERNEL(test_equal);

TEST_SUITE(equal)
{
  TEST_HOSTDEV(test_equal);
}

auto
main() -> int
{
  RUN_SUITE(equal);
  return 0;
}
