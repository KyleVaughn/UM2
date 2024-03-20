#include <um2/stdlib/numeric/iota.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(iota_int)
{
  constexpr int n = 10;
  int a[n] = {0};
  um2::iota(a, a + n, 1);
  for (int i = 0; i < n; ++i) {
    ASSERT(a[i] == i + 1);
  }
  um2::iota(a, a + n, 0);
  for (int i = 0; i < n; ++i) {
    ASSERT(a[i] == i);
  }
}
MAKE_CUDA_KERNEL(iota_int);

TEST_SUITE(iota) { TEST_HOSTDEV(iota_int); }

auto
main() -> int
{
  RUN_SUITE(iota);
  return 0;
}
