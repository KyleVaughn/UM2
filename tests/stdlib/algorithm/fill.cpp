#include <um2/stdlib/algorithm/fill.hpp>

#include "../../test_macros.hpp"

template <typename T>
HOSTDEV
TEST_CASE(fill_test)
{
  T a[10] = {0};
  um2::fill(&a[0], &a[0] + 10, 1);
  for (auto const & i : a) {
    ASSERT(i == 1);
  }
  um2::fill(&a[0], &a[0] + 10, 2);
  for (auto const & i : a) {
    ASSERT(i == 2);
  }
}

#if UM2_USE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(fill_test, T);
#endif

template <typename T>
TEST_SUITE(fill)
{
  TEST_HOSTDEV(fill_test, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE(fill<int32_t>);
  RUN_SUITE(fill<int64_t>);
  return 0;
}
