#include <um2/stdlib/algorithm.hpp>

#include "../test_macros.hpp"

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

HOSTDEV
TEST_CASE(copy_nontrivial)
{
  struct A {
    int a;
    int b;
    A()
        : a(0),
          b(0)
    {
    }
    HOSTDEV
    A(int aa, int bb)
        : a(aa),
          b(bb)
    {
    }
    HOSTDEV
    A(A const & other) = default;

    HOSTDEV
    A(A && other)
    noexcept
        : a(other.a),
          b(other.b)
    {
    }
    HOSTDEV auto
    operator=(A const & other) -> A &
    {
      if (this == &other) {
        return *this;
      }
      a = other.a;
      b = other.b;
      return *this;
    }
    HOSTDEV auto
    operator=(A && other) noexcept -> A &
    {
      a = other.a;
      b = other.b;
      return *this;
    }
    ~A() = default;
  };
  A a[5] = {A(0, 0), A(1, 1), A(2, 2), A(3, 3), A(4, 4)};
  A b[5] = {A(0, 0), A(0, 0), A(0, 0), A(0, 0), A(0, 0)};
  um2::copy(&a[0], &a[0] + 5, &b[0]);
  for (int i = 0; i < 5; ++i) {
    ASSERT(a[i].a == b[i].a);
    ASSERT(a[i].b == b[i].b);
  }
}

//=============================================================================
// fill
//=============================================================================

HOSTDEV
TEST_CASE(fill_int)
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

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(clamp_int);
MAKE_CUDA_KERNEL(clamp_float);

MAKE_CUDA_KERNEL(copy_trivial);
MAKE_CUDA_KERNEL(copy_nontrivial);

MAKE_CUDA_KERNEL(fill_int);

MAKE_CUDA_KERNEL(maxmin_int);
MAKE_CUDA_KERNEL(maxmin_float);
#endif

TEST_SUITE(clamp)
{
  TEST_HOSTDEV(clamp_int);
  TEST_HOSTDEV(clamp_float);
}

TEST_SUITE(copy)
{
  TEST_HOSTDEV(copy_trivial);
  TEST_HOSTDEV(copy_nontrivial);
}

TEST_SUITE(fill) { TEST_HOSTDEV(fill_int); }

TEST_SUITE(maxmin)
{
  TEST_HOSTDEV(maxmin_int);
  TEST_HOSTDEV(maxmin_float);
}

auto
main() -> int
{
  RUN_SUITE(clamp);
  RUN_SUITE(copy);
  RUN_SUITE(fill);
  RUN_SUITE(maxmin);
  return 0;
}
