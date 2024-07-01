#include <um2/config.hpp>
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
  static_assert(um2::equal(a, a, c));
}

HOSTDEV
constexpr auto
foo() -> bool
{
  int constexpr a[] = {10, 20, 30};
  int constexpr b[] = {10, 20, 30};
  return um2::equal(a, a + 3, b);
}

HOSTDEV
TEST_CASE(test_equal_constexpr) { static_assert(foo()); }

MAKE_CUDA_KERNEL(test_equal);
MAKE_CUDA_KERNEL(test_equal_constexpr);

TEST_SUITE(equal)
{
  TEST_HOSTDEV(test_equal);
  TEST_HOSTDEV(test_equal_constexpr);
}

auto
main() -> int
{
  RUN_SUITE(equal);
  return 0;
}
