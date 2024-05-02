#include <um2/stdlib/cstring/memcmp.hpp>

#include "../../test_macros.hpp"

// clang-tidy seems to crash when the test suite is named memcmp...

HOSTDEV
TEST_CASE(test_memcmp)
{
  constexpr const char * s1 = "abc";
  constexpr const char * s2 = "abc";
  constexpr const char * s3 = "bcd";

  STATIC_ASSERT(um2::memcmp(s1, s2, 0) == 0);
  STATIC_ASSERT(um2::memcmp(s1, s2, 1) == 0);
  STATIC_ASSERT(um2::memcmp(s1, s2, 3) == 0);
  STATIC_ASSERT(um2::memcmp(s1, s3, 2) < 0);
  STATIC_ASSERT(um2::memcmp(s3, s1, 2) > 0);
}

MAKE_CUDA_KERNEL(test_memcmp);

TEST_SUITE(um2_memcmp) { TEST_HOSTDEV(test_memcmp); }

auto
main() -> int
{
  RUN_SUITE(um2_memcmp);
  return 0;
}
