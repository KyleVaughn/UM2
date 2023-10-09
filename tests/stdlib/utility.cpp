#include <um2/stdlib/utility.hpp>

#include "../test_macros.hpp"

//=============================================================================
// swap
//=============================================================================

HOSTDEV
TEST_CASE(swap_int)
{
  int a = 1;
  int b = 2;
  um2::swap(a, b);
  ASSERT(a == 2);
  ASSERT(b == 1);
}

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(swap_int);
#endif

TEST_SUITE(swap) { TEST_HOSTDEV(swap_int); }

auto
main() -> int
{
  RUN_SUITE(swap);
  return 0;
}
