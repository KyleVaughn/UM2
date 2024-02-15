#include <um2/stdlib/algorithm/fill.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(fill_test)
{
  int a[10] = {0};
  um2::fill(&a[0], &a[0] + 10, 1);
  for (auto const & i : a) {
    ASSERT(i == 1);
  }
  um2::fill(&a[0], &a[0] + 10, 2);
  for (auto const & i : a) {
    ASSERT(i == 2);
  }
}
MAKE_CUDA_KERNEL(fill_test);

TEST_SUITE(fill)
{
  TEST_HOSTDEV(fill_test);
}
auto
main() -> int
{
  RUN_SUITE(fill);
  return 0;
}
