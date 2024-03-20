#include <um2/stdlib/cstring/strlen.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(test_strlen)
{
  static_assert(um2::strlen("") == 0);
  static_assert(um2::strlen("a") == 1);
  static_assert(um2::strlen("ab") == 2);
  static_assert(um2::strlen("abc") == 3);
}

MAKE_CUDA_KERNEL(test_strlen);

TEST_SUITE(strlen) { TEST_HOSTDEV(test_strlen); }

auto
main() -> int
{
  RUN_SUITE(strlen);
  return 0;
}
