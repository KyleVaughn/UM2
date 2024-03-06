#include <um2/common/permutation.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(sortPermutation)
{
  um2::Vector<T> const v = {5, 3, 1, 4, 2};
  um2::Vector<Int> perm(v.size());
  um2::sortPermutation(v.cbegin(), v.cend(), perm.begin());
  um2::Vector<T> sorted_v(v.size());
  for (Int i = 0; i < v.size(); ++i) {
    sorted_v[i] = v[perm[i]];
  }
  ASSERT(std::is_sorted(sorted_v.cbegin(), sorted_v.cend()));
  um2::Vector<Int> const expected_perm = {2, 4, 1, 3, 0};
  ASSERT(perm == expected_perm);
}

template <typename T>
TEST_CASE(applyPermutation)
{
  um2::Vector<T> v = {5, 3, 1, 4, 2};
  um2::Vector<Int> const perm = {2, 4, 1, 3, 0};
  um2::applyPermutation(v.begin(), v.end(), perm.cbegin());
  um2::Vector<T> const expected_v = {1, 2, 3, 4, 5};
  ASSERT(v == expected_v);
}

template <typename T>
TEST_CASE(invertPermutation)
{
  um2::Vector<T> const v = {5, 3, 1, 4, 2};
  um2::Vector<T> v2 = v;
  um2::Vector<Int> const perm = {2, 4, 1, 3, 0};
  um2::Vector<Int> inv_perm(5);
  um2::invertPermutation(perm.cbegin(), perm.cend(), inv_perm.begin());
  um2::applyPermutation(v2.begin(), v2.end(), perm.cbegin());
  um2::Vector<T> const expected_v = {1, 2, 3, 4, 5};
  ASSERT(v2 == expected_v);
  um2::applyPermutation(v2.begin(), v2.end(), inv_perm.cbegin());
  ASSERT(v2 == v);
}

TEST_SUITE(permutation)
{
  TEST((sortPermutation<int>));
  TEST((sortPermutation<float>));
  TEST((applyPermutation<int>));
  TEST((invertPermutation<int>));
}

auto
main() -> int
{
  RUN_SUITE(permutation);
  return 0;
}
