#include <um2/stdlib/algorithm/copy.hpp>

#include "../../test_macros.hpp"

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
MAKE_CUDA_KERNEL(copy_trivial)

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct A {
  int a;
  int * b{nullptr};

  HOSTDEV A() = default;

  HOSTDEV
  A(A const & other) 
    : a(other.a)
  {
  }

  HOSTDEV
  auto operator=(A const & other) -> A & 
  {
    if (this != &other) {
      a = other.a;
    }
    return *this;
  } 
};

HOSTDEV
TEST_CASE(copy_nontrivial)
{
  A a[10];
  A b[10];
  // Initialize a to {i, * to i}
  for (int i = 0; i < 10; ++i) {
    a[i].a = i;
    a[i].b = &a[i].a;
  }
  um2::copy(&a[0], &a[0] + 10, &b[0]);
  for (int i = 0; i < 10; ++i) {
    ASSERT(b[i].a == i);
    ASSERT(b[i].b == nullptr);
  }
}
MAKE_CUDA_KERNEL(copy_nontrivial)

TEST_SUITE(copy) 
{ 
  TEST_HOSTDEV(copy_trivial); 
  TEST_HOSTDEV(copy_nontrivial);
}

auto
main() -> int
{
  RUN_SUITE(copy);
  return 0;
}
