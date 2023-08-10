#include <um2/stdlib/algorithm/copy.hpp>

#include "../../test_macros.hpp"

template <typename T>
HOSTDEV
TEST_CASE(copy_trivial)
{
  T a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  T b[10] = {0};
  um2::copy(&a[0], &a[0] + 10, &b[0]);
  for (int i = 0; i < 10; ++i) {
    ASSERT(a[i] == b[i]);
  }
}

template <typename T>
HOSTDEV
TEST_CASE(copy_nontrivial)
{
  struct A {
    T a;
    T b;
    A()
        : a(0),
          b(0)
    {
    }
    HOSTDEV
    A(T aa, T bb)
        : a(aa),
          b(bb)
    {
    }
    HOSTDEV
    A(A const & other)
        : a(other.a),
          b(other.b)
    {
    }
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
    HOSTDEV ~A() = default;
  };
  A a[5] = {A(0, 0), A(1, 1), A(2, 2), A(3, 3), A(4, 4)};
  A b[5] = {A(0, 0), A(0, 0), A(0, 0), A(0, 0), A(0, 0)};
  um2::copy(&a[0], &a[0] + 5, &b[0]);
  for (int i = 0; i < 5; ++i) {
    ASSERT(a[i].a == b[i].a);
    ASSERT(a[i].b == b[i].b);
  }
}

#if UM2_ENABLE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(copy_trivial, T);

template <typename T>
MAKE_CUDA_KERNEL(copy_nontrivial, T);
#endif

template <typename T>
TEST_SUITE(copy_suite)
{
  TEST_HOSTDEV(copy_trivial, 1, 1, T);
  TEST_HOSTDEV(copy_nontrivial, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE(copy_suite<int64_t>);
  return 0;
}
