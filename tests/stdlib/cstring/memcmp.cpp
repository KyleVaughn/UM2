#include <um2/stdlib/cstring/memcmp.hpp>

#include "../../test_macros.hpp"

// clang-tidy seems to crash when the test suite is named memcmp...

HOSTDEV
TEST_CASE(test_memcmp)
{
  char const * const s1 = "abc";
  char const * const s2 = "abc";
  char const * const s3 = "bcd";

  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT(um2::memcmp(s1, s2, 0) == 0);
  ASSERT(um2::memcmp(s1, s2, 1) == 0);
  ASSERT(um2::memcmp(s1, s2, 3) == 0);
  ASSERT(um2::memcmp(s1, s3, 2) < 0);
  ASSERT(um2::memcmp(s3, s1, 2) > 0);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

MAKE_CUDA_KERNEL(test_memcmp);

TEST_SUITE(um2_memcmp) { TEST_HOSTDEV(test_memcmp); }

auto
main() -> int
{
  RUN_SUITE(um2_memcmp);
  return 0;
}
