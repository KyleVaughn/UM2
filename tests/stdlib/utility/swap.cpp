#include <um2/config.hpp>
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

HOSTDEV
auto constexpr foo()
{
  int a = 1;
  int b = 2;
  um2::swap(a, b);
  return b;
}

HOSTDEV
TEST_CASE(swap_constexpr) { static_assert(foo() == 1); }
MAKE_CUDA_KERNEL(swap_constexpr);

TEST_SUITE(test_swap)
{
  TEST_HOSTDEV(swap_int);
  TEST_HOSTDEV(swap_array);
  TEST_HOSTDEV(swap_constexpr);
}

auto
main() -> int
{
  RUN_SUITE(test_swap);
  return 0;
}
