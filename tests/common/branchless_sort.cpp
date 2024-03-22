#include <um2/common/branchless_sort.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>

#include "../test_macros.hpp"
#include <iostream>

#define TEST_SORT3(X, Y, Z) \
{ \
  *x = X; \
  *y = Y; \
  *z = Z; \
  um2::sort3(x, y, z); \
  ASSERT(*x <= *y); \
  ASSERT(*y <= *z); \
}

template <typename T>
HOSTDEV
TEST_CASE(sort3)
{
  T arr[3];
  T * const x = &arr[0]; 
  T * const y = &arr[1];
  T * const z = &arr[2];
  // All permutations of 1, 2, 3
  TEST_SORT3(1, 2, 3);
  TEST_SORT3(1, 3, 2);
  TEST_SORT3(2, 1, 3);
  TEST_SORT3(2, 3, 1);
  TEST_SORT3(3, 1, 2);
  TEST_SORT3(3, 2, 1)
}

template <typename T>
TEST_SUITE(branchless_sort)
{
  TEST_HOSTDEV(sort3, T)
}

auto
main() -> int
{
  RUN_SUITE(branchless_sort<int32_t>);
  RUN_SUITE(branchless_sort<double>);
  return 0;
}
