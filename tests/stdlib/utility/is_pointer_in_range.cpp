#include <um2/config.hpp>
#include <um2/stdlib/utility/is_pointer_in_range.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(test_is_pointer_in_range)
{
  int a[4] = {1, 2, 3, 4};
  int * const begin = a;
  int * const end = a + 4;
  ASSERT(um2::is_pointer_in_range(begin, end, a));
  ASSERT(um2::is_pointer_in_range(begin, end, a + 1));
  ASSERT(um2::is_pointer_in_range(begin, end, a + 2));
  ASSERT(!um2::is_pointer_in_range(begin, end, a + 4));
// CUDA warns about out-of-bounds pointer. This is what we want to check. Suppress.
#pragma nv_diagnostic push
#pragma nv_diag_suppress 170
  ASSERT(!um2::is_pointer_in_range(begin, end, a - 1));
#pragma nv_diagnostic pop
}
MAKE_CUDA_KERNEL(test_is_pointer_in_range);

HOSTDEV
TEST_CASE(test_is_pointer_in_range_constexpr)
{
  int constexpr a[4] = {1, 2, 3, 4};
  static_assert(um2::is_pointer_in_range(a, a + 4, a));
  static_assert(um2::is_pointer_in_range(a, a + 4, a + 1));
  static_assert(um2::is_pointer_in_range(a, a + 4, a + 2));
  static_assert(!um2::is_pointer_in_range(a, a + 4, a + 4));
}
MAKE_CUDA_KERNEL(test_is_pointer_in_range_constexpr)

TEST_SUITE(is_pointer_in_range)
{
  TEST_HOSTDEV(test_is_pointer_in_range);
  TEST_HOSTDEV(test_is_pointer_in_range_constexpr);
}

auto
main() -> int
{
  RUN_SUITE(is_pointer_in_range);
  return 0;
}
