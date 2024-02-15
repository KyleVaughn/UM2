#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/utility/swap.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(is_sorted_int)
{
  int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 8};
  ASSERT(!um2::is_sorted(&a[0], &a[0] + 10));
  um2::swap(a[8], a[9]);
  ASSERT(um2::is_sorted(&a[0], &a[0] + 10));
}
MAKE_CUDA_KERNEL(is_sorted_int);


TEST_SUITE(is_sorted) { TEST_HOSTDEV(is_sorted_int); }

auto
main() -> int
{
  RUN_SUITE(is_sorted);
  return 0;
}
