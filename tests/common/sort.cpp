#include <um2/common/sort.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

HOSTDEV
TEST_CASE(insertionSort)
{
  int a[9] = {3, 7, 4, 9, 5, 2, 6, 1, 8};
  um2::insertionSort(&a[0], &a[0] + 9);
  for (int i = 1; i < 10; ++i) {
    ASSERT(a[i - 1] == i);
  }
}
MAKE_CUDA_KERNEL(insertionSort);

template <typename T>
TEST_CASE(sortPermutation)
{
  um2::Vector<T> const v = {5, 3, 1, 4, 2};
  um2::Vector<Size> perm(v.size());
  um2::sortPermutation(v.cbegin(), v.cend(), perm.begin());
  um2::Vector<T> sorted_v(v.size());
  for (Size i = 0; i < v.size(); ++i) {
    sorted_v[i] = v[perm[i]];
  }
  ASSERT(std::is_sorted(sorted_v.cbegin(), sorted_v.cend()));
  um2::Vector<Size> const expected_perm = {2, 4, 1, 3, 0};
  ASSERT(perm == expected_perm);
}

template <typename T>
TEST_CASE(applyPermutation)
{
  um2::Vector<T> v = {5, 3, 1, 4, 2};
  um2::Vector<Size> const perm = {2, 4, 1, 3, 0};
  applyPermutation(v, perm);
  um2::Vector<T> const expected_v = {1, 2, 3, 4, 5};
  ASSERT(v == expected_v);
}

TEST_SUITE(sort)
{
  TEST((sortPermutation<int>));
  TEST((sortPermutation<float>));
  TEST((applyPermutation<int>));
  TEST((applyPermutation<float>));
  TEST_HOSTDEV(insertionSort, 1, 1);
}

auto
main() -> int
{
  RUN_SUITE(sort);
  return 0;
}
