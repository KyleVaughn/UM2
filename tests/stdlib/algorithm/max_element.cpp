#include <um2/config.hpp>
#include <um2/stdlib/algorithm/max_element.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(max_element_int)
{
  int constexpr a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 8};
  STATIC_ASSERT(um2::max_element(&a[0], &a[0] + 10) == &a[8]);

  int constexpr b[10] = {0, 2, 4, 3, 4, 5, 9, 9, 7, 8};
  STATIC_ASSERT(um2::max_element(&b[0], &b[0] + 10) == &b[6]);
}
MAKE_CUDA_KERNEL(max_element_int);

TEST_SUITE(max_element) { TEST_HOSTDEV(max_element_int); }

auto
main() -> int
{
  RUN_SUITE(max_element);
  return 0;
}
