#include <um2/config.hpp>
#include <um2/stdlib/algorithm/lexicographical_compare.hpp>

#include "../../test_macros.hpp"

HOSTDEV
TEST_CASE(test_lexicographical_compare)
{
  int constexpr a[] = {1, 2, 3};
  int constexpr b[] = {1, 2, 3};
  int constexpr c[] = {3, 2, 1};

  STATIC_ASSERT(!um2::lexicographical_compare(a, a + 3, b, b + 3));
  STATIC_ASSERT(um2::lexicographical_compare(a, a + 3, c, c + 3));
}

MAKE_CUDA_KERNEL(test_lexicographical_compare);

TEST_SUITE(lexicographical_compare) { TEST_HOSTDEV(test_lexicographical_compare); }

auto
main() -> int
{
  RUN_SUITE(lexicographical_compare);
  return 0;
}
