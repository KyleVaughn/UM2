#include <um2/common/branchless_sort.hpp>
#include <um2/config.hpp>

#include <cstdint>

#include "../test_macros.hpp"

#define TEST_SORT3(X, Y, Z)                                                              \
  {                                                                                      \
    *x = X;                                                                              \
    *y = Y;                                                                              \
    *z = Z;                                                                              \
    um2::sort3(x, y, z);                                                                 \
    ASSERT(*x <= *y);                                                                    \
    ASSERT(*y <= *z);                                                                    \
  }

#define TEST_SORT4(X1, X2, X3, X4)                                                       \
  {                                                                                      \
    *x1 = X1;                                                                            \
    *x2 = X2;                                                                            \
    *x3 = X3;                                                                            \
    *x4 = X4;                                                                            \
    um2::sort4(x1, x2, x3, x4);                                                          \
    ASSERT(*x1 <= *x2);                                                                  \
    ASSERT(*x2 <= *x3);                                                                  \
    ASSERT(*x3 <= *x4);                                                                  \
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
HOSTDEV
TEST_CASE(sort4)
{
  T arr[4];
  T * const x1 = &arr[0];
  T * const x2 = &arr[1];
  T * const x3 = &arr[2];
  T * const x4 = &arr[3];
  // All permutations of 1, 2, 3, 4
  TEST_SORT4(1, 2, 3, 4);
  TEST_SORT4(1, 2, 4, 3);
  TEST_SORT4(1, 3, 2, 4);
  TEST_SORT4(1, 3, 4, 2);
  TEST_SORT4(1, 4, 2, 3);
  TEST_SORT4(1, 4, 3, 2);
  TEST_SORT4(2, 1, 3, 4);
  TEST_SORT4(2, 1, 4, 3);
  TEST_SORT4(2, 3, 1, 4);
  TEST_SORT4(2, 3, 4, 1);
  TEST_SORT4(2, 4, 1, 3);
  TEST_SORT4(2, 4, 3, 1);
  TEST_SORT4(3, 1, 2, 4);
  TEST_SORT4(3, 1, 4, 2);
  TEST_SORT4(3, 2, 1, 4);
  TEST_SORT4(3, 2, 4, 1);
  TEST_SORT4(3, 4, 1, 2);
  TEST_SORT4(3, 4, 2, 1);
  TEST_SORT4(4, 1, 2, 3);
  TEST_SORT4(4, 1, 3, 2);
  TEST_SORT4(4, 2, 1, 3);
  TEST_SORT4(4, 2, 3, 1);
  TEST_SORT4(4, 3, 1, 2);
  TEST_SORT4(4, 3, 2, 1);
}

#if UM2_USE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(sort3, T);

template <typename T>
MAKE_CUDA_KERNEL(sort4, T);
#endif

template <typename T>
TEST_SUITE(branchless_sort)
{
  TEST_HOSTDEV(sort3, T)
  TEST_HOSTDEV(sort4, T)
}

auto
main() -> int
{
  RUN_SUITE(branchless_sort<int32_t>);
  RUN_SUITE(branchless_sort<double>);
  return 0;
}
