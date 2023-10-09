#include <um2/stdlib/numeric.hpp>

#include "../test_macros.hpp"

HOSTDEV
TEST_CASE(iota)
{
  constexpr int n = 10;
  int a[n] = {0};
  um2::iota(a, a + n, 1);
  for (int i = 0; i < n; ++i) {
    assert(a[i] == i + 1);
  }
}
MAKE_CUDA_KERNEL(iota);

TEST_SUITE(iota_suite) { TEST_HOSTDEV(iota); }

auto
main() -> int
{
  RUN_SUITE(iota_suite);
  return 0;
}
