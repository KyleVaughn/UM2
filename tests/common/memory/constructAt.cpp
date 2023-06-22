#include "../../test_macros.hpp"
#include <um2/common/memory/constructAt.hpp>

#include <cstdlib>

// NOLINTBEGIN
#ifndef __CUDA_ARCH__
int count = 0;
#else
DEVICE int count = 0;
#endif
struct Counted {
  HOSTDEV
  Counted() { ++count; }
  HOSTDEV
  Counted(Counted const &) { ++count; }
  HOSTDEV ~Counted() { --count; }
  HOSTDEV friend void operator&(Counted) = delete;
};

struct VCounted {
  HOSTDEV
  VCounted() { ++count; }
  HOSTDEV
  VCounted(VCounted const &) { ++count; }
  HOSTDEV virtual ~VCounted() { --count; }
  HOSTDEV friend void operator&(VCounted) = delete;
};

struct DCounted : VCounted {
  HOSTDEV friend void operator&(DCounted) = delete;
};
// NOLINTEND

// ------------------------------------------------------------
// destroyAt
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(test_destroyAt)
{
  {
    void * mem1 = malloc(sizeof(Counted));
    void * mem2 = malloc(sizeof(Counted));
    assert(mem1 != nullptr);
    assert(mem2 != nullptr);
    assert(count == 0);
    Counted * ptr1 = nullptr;
    ptr1 = ::new (mem1) Counted();
    assert(ptr1 != nullptr);
    Counted * ptr2 = nullptr;
    ptr2 = ::new (mem2) Counted();
    assert(ptr2 != nullptr);
    assert(count == 2);
    um2::destroyAt(ptr1);
    assert(count == 1);
    um2::destroyAt(ptr2);
    assert(count == 0);
    free(mem1);
    free(mem2);
    count = 0;
  }
  {
    void * mem1 = malloc(sizeof(DCounted));
    void * mem2 = malloc(sizeof(DCounted));
    assert(mem1 != nullptr);
    assert(mem2 != nullptr);
    assert(count == 0);
    DCounted * ptr1 = nullptr;
    ptr1 = ::new (mem1) DCounted();
    assert(ptr1 != nullptr);
    DCounted * ptr2 = nullptr;
    ptr2 = ::new (mem2) DCounted();
    assert(ptr2 != nullptr);
    assert(count == 2);
    um2::destroyAt(ptr1);
    assert(count == 1);
    um2::destroyAt(ptr2);
    assert(count == 0);
    free(mem1);
    free(mem2);
  }
}
MAKE_CUDA_KERNEL(test_destroyAt);

// ------------------------------------------------------------
// constructAt
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(test_constructAt)
{
  struct S {
    int x;
    float y;
    double z;

    HOSTDEV
    S(int x_in, float y_in, double z_in)
        : x(x_in),
          y(y_in),
          z(z_in)
    {
    }
  };

  alignas(S) unsigned char storage[sizeof(S)];

  S * ptr = um2::constructAt(reinterpret_cast<S *>(storage), 42, 2.71828F, 3.1415);
  assert((*ptr).x == 42);
  assert(((*ptr).y - 2.71828F) < 0.0001F);
  assert(((*ptr).z - 3.1415) < 0.0001);
  um2::destroyAt(ptr);
}
MAKE_CUDA_KERNEL(test_constructAt);

// ------------------------------------------------------------
// destroy
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(test_destroy)
{
  {
    void * mem = malloc(5 * sizeof(Counted));
    assert(mem != nullptr);
    assert(count == 0);
    Counted * ptr_begin = nullptr;
    ptr_begin = ::new (mem) Counted();
    // Initialize the rest of the memory.
    for (size_t i = 1; i < 5; ++i) {
      void * mem_init =
          static_cast<void *>(static_cast<char *>(mem) + i * sizeof(Counted));
      ::new (mem_init) Counted();
    }
    assert(ptr_begin != nullptr);
    Counted * ptr_end = ptr_begin + 5;
    assert(count == 5);
    um2::destroy(ptr_begin + 2, ptr_end);
    assert(count == 2);
    um2::destroy(ptr_begin, ptr_begin + 2);
    assert(count == 0);
    free(mem);
  }
}
MAKE_CUDA_KERNEL(test_destroy);

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

TEST_SUITE(constructAt)
{
  TEST_HOSTDEV(test_destroyAt);
  TEST_HOSTDEV(test_constructAt);
  TEST_HOSTDEV(test_destroy);
}

auto
main() -> int
{
  RUN_TESTS(constructAt);
  return 0;
}
