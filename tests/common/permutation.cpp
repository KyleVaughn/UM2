#include <um2/common/permutation.hpp>
#include <um2/stdlib/Vector.hpp>

#include "../test_macros.hpp"

template <typename T>
TEST_CASE(sortPermutation)
{
  um2::Vector<T> const v{5, 3, 1, 4, 2};
  um2::Vector<Size> perm(v.size());
  um2::sortPermutation(v.cbegin(), v.cend(), perm.begin());
  um2::Vector<T> sorted_v(v.size());
  for (Size i = 0; i < v.size(); ++i) {
    sorted_v[i] = v[perm[i]];
  }
  // cppcheck-suppress assertWithSideEffect; justification: no side effects
  ASSERT(std::is_sorted(sorted_v.begin(), sorted_v.end()));
  um2::Vector<Size> const expected_perm{2, 4, 1, 3, 0};
  ASSERT(perm == expected_perm);
}

// template <typename T>
// TEST_CASE(applyPermutation)
//{
//   um2::Vector<T> v{5, 3, 1, 4, 2};
//   um2::Vector<Size> const perm{2, 4, 1, 3, 0};
//   applyPermutation(v, perm);
//   um2::Vector<T> const expected_v{1, 2, 3, 4, 5};
//   ASSERT(v == expected_v);
// }

//==============================================================================
// CUDA
//==============================================================================
// #if UM2_USE_CUDA
// #endif // UM2_USE_CUDA

template <class T>
TEST_SUITE(sort)
{
  TEST((sortPermutation<T>))
  //  TEST((applyPermutation<T>))
}

auto
main() -> int
{
  RUN_SUITE(sort<int32_t>);
  RUN_SUITE(sort<uint32_t>);
  RUN_SUITE(sort<int64_t>);
  RUN_SUITE(sort<uint64_t>);
  RUN_SUITE(sort<float>);
  RUN_SUITE(sort<double>);
  return 0;
}
