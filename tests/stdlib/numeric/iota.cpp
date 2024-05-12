#include <um2/config.hpp>
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

constexpr auto
foo() -> int
{
  int a[10] = {0};
  um2::iota(a, a + 10, 1);
  int sum = 0;
  for (auto const i : a) {
    sum += i;
  }
  return sum;
}

HOSTDEV
TEST_CASE(iota_int_constexpr) { STATIC_ASSERT(foo() == 55); }
MAKE_CUDA_KERNEL(iota_int_constexpr);

TEST_SUITE(iota)
{
  TEST_HOSTDEV(iota_int);
  TEST_HOSTDEV(iota_int_constexpr);
}

auto
main() -> int
{
  RUN_SUITE(iota);
  return 0;
}
