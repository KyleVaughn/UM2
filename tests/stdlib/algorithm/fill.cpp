#include <um2/stdlib/algorith/fill.hpp>

#include "../test_macros.hpp"

//=============================================================================
// fill_n
//=============================================================================

HOSTDEV
TEST_CASE(fill_n)
{
  int a[10] = {0};
  int * p = um2::fill_n(&a[0], 5, 1);
  ASSERT(p == &a[5]);
  for (int i = 0; i < 5; ++i) {
    ASSERT(a[i] == 1);
  }
  for (int i = 5; i < 10; ++i) {
    ASSERT(a[i] == 0);
  }
}
MAKE_CUDA_KERNEL(fill_n);

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
  TEST_HOSTDEV(fill_n);
  TEST_HOSTDEV(fill_test);
}
auto
main() -> int
{
  RUN_SUITE(fill);
  return 0;
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)
