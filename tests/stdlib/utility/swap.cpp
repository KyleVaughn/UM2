#include <um2/stdlib/utility/swap.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(swap_int)
{
  int a = 1;
  int b = 2;
  um2::swap(a, b);
  ASSERT(a == 2);
  ASSERT(b == 1);
}
MAKE_CUDA_KERNEL(swap_int);

HOSTDEV
TEST_CASE(swap_array)
{
  int a[2] = {1, 2};
  int b[2] = {3, 4};
  um2::swap(a, b);
  ASSERT(a[0] == 3);
  ASSERT(a[1] == 4);
  ASSERT(b[0] == 1);
  ASSERT(b[1] == 2);
}
MAKE_CUDA_KERNEL(swap_array);

TEST_SUITE(swap)
{
  TEST(swap_int);
  TEST(swap_array);
}

auto
main() -> int
{
  RUN_SUITE(swap);
  return 0;
}
