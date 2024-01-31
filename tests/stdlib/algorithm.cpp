#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/utility.hpp>

#include "../test_macros.hpp"

// NOLINTBEGIN(cert-dcl03-c,misc-static-assert); justification: compiler-dependent

//=============================================================================
// clamp
//=============================================================================

HOSTDEV
TEST_CASE(clamp_int)
{
  ASSERT(um2::clamp(0, 1, 3) == 1);
  ASSERT(um2::clamp(1, 1, 3) == 1);
  ASSERT(um2::clamp(2, 1, 3) == 2);
  ASSERT(um2::clamp(3, 1, 3) == 3);
  ASSERT(um2::clamp(4, 1, 3) == 3);
}

HOSTDEV
TEST_CASE(clamp_float)
{
  ASSERT_NEAR(um2::clamp(0.0F, 1.0F, 3.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::clamp(1.0F, 1.0F, 3.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::clamp(2.0F, 1.0F, 3.0F), 2.0F, 1e-6F);
  ASSERT_NEAR(um2::clamp(3.0F, 1.0F, 3.0F), 3.0F, 1e-6F);
  ASSERT_NEAR(um2::clamp(4.0F, 1.0F, 3.0F), 3.0F, 1e-6F);
}

//=============================================================================
// copy
//=============================================================================

HOSTDEV
TEST_CASE(copy_trivial)
{
  int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int b[10] = {0};
  um2::copy(&a[0], &a[0] + 10, &b[0]);
  for (int i = 0; i < 10; ++i) {
    ASSERT(a[i] == b[i]);
  }
}

//=============================================================================
// fill_n
//=============================================================================

HOSTDEV
TEST_CASE(fill_n)
{
  int a[10] = {0};
  int * p = um2::fill_n(&a[0], 5, 1);
  ASSERT(p == &a[5]);
  for (int i = 0; i < 5; ++i) {
    ASSERT(a[i] == 1);
  }
  for (int i = 5; i < 10; ++i) {
    ASSERT(a[i] == 0);
  }
}

//=============================================================================
// fill
//=============================================================================

HOSTDEV
TEST_CASE(fill_test)
{
  int a[10] = {0};
  um2::fill(&a[0], &a[0] + 10, 1);
  for (auto const & i : a) {
    ASSERT(i == 1);
  }
  um2::fill(&a[0], &a[0] + 10, 2);
  for (auto const & i : a) {
    ASSERT(i == 2);
  }
}

//=============================================================================
// is_sorted
//=============================================================================

HOSTDEV
TEST_CASE(is_sorted_int)
{
  int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 8};
  ASSERT(!um2::is_sorted(&a[0], &a[0] + 10));
  um2::swap(a[8], a[9]);
  ASSERT(um2::is_sorted(&a[0], &a[0] + 10));
}

//=============================================================================
// min/max
//=============================================================================

HOSTDEV
TEST_CASE(maxmin_int)
{
  ASSERT(um2::min(0, 1) == 0);
  ASSERT(um2::min(1, 0) == 0);
  ASSERT(um2::min(0, 0) == 0);
  ASSERT(um2::max(0, 1) == 1);
  ASSERT(um2::max(1, 0) == 1);
  ASSERT(um2::max(0, 0) == 0);
}

HOSTDEV
TEST_CASE(maxmin_float)
{
  ASSERT_NEAR(um2::min(0.0F, 1.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::min(1.0F, 0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::min(0.0F, 0.0F), 0.0F, 1e-6F);
  ASSERT_NEAR(um2::max(0.0F, 1.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::max(1.0F, 0.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::max(0.0F, 0.0F), 0.0F, 1e-6F);
}

//=============================================================================
// max_element
//=============================================================================

HOSTDEV
TEST_CASE(max_element_int)
{
  int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 8};
  ASSERT(um2::max_element(&a[0], &a[0] + 10) == &a[8]);

  int b[10] = {0, 2, 4, 3, 4, 5, 9, 9, 7, 8};
  ASSERT(um2::max_element(&b[0], &b[0] + 10) == &b[6]);
}

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(clamp_int);
MAKE_CUDA_KERNEL(clamp_float);

MAKE_CUDA_KERNEL(copy_trivial);

MAKE_CUDA_KERNEL(fill_int);

MAKE_CUDA_KERNEL(is_sorted_int);

MAKE_CUDA_KERNEL(maxmin_int);
MAKE_CUDA_KERNEL(maxmin_float);

MAKE_CUDA_KERNEL(max_element_int);
#endif

TEST_SUITE(clamp)
{
  TEST_HOSTDEV(clamp_int);
  TEST_HOSTDEV(clamp_float);
}

TEST_SUITE(copy) { TEST_HOSTDEV(copy_trivial); }

TEST_SUITE(fill)
{
  TEST_HOSTDEV(fill_n);
  TEST_HOSTDEV(fill_test);
}

TEST_SUITE(is_sorted) { TEST_HOSTDEV(is_sorted_int); }

TEST_SUITE(maxmin)
{
  TEST_HOSTDEV(maxmin_int);
  TEST_HOSTDEV(maxmin_float);
}

TEST_SUITE(max_element) { TEST_HOSTDEV(max_element_int); }

auto
main() -> int
{
  RUN_SUITE(clamp);
  RUN_SUITE(copy);
  RUN_SUITE(fill);
  RUN_SUITE(is_sorted);
  RUN_SUITE(maxmin);
  RUN_SUITE(max_element);
  return 0;
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)
