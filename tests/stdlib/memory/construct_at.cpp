#include <um2/stdlib/memory/construct_at.hpp>

#include <cstdlib>

#include "../../test_macros.hpp"

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
// destroy_at
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(destroy_at)
{
  {
    void * mem1 = malloc(sizeof(Counted));
    void * mem2 = malloc(sizeof(Counted));
    ASSERT(mem1 != nullptr);
    ASSERT(mem2 != nullptr);
    ASSERT(count == 0);
    Counted * ptr1 = nullptr;
    ptr1 = ::new (mem1) Counted();
    ASSERT(ptr1 != nullptr);
    Counted * ptr2 = nullptr;
    ptr2 = ::new (mem2) Counted();
    ASSERT(ptr2 != nullptr);
    ASSERT(count == 2);
    um2::destroy_at(ptr1);
    ASSERT(count == 1);
    um2::destroy_at(ptr2);
    ASSERT(count == 0);
    free(mem1);
    free(mem2);
    count = 0;
  }
  {
    void * mem1 = malloc(sizeof(DCounted));
    void * mem2 = malloc(sizeof(DCounted));
    ASSERT(mem1 != nullptr);
    ASSERT(mem2 != nullptr);
    ASSERT(count == 0);
    DCounted * ptr1 = nullptr;
    ptr1 = ::new (mem1) DCounted();
    ASSERT(ptr1 != nullptr);
    DCounted * ptr2 = nullptr;
    ptr2 = ::new (mem2) DCounted();
    ASSERT(ptr2 != nullptr);
    ASSERT(count == 2);
    um2::destroy_at(ptr1);
    ASSERT(count == 1);
    um2::destroy_at(ptr2);
    ASSERT(count == 0);
    free(mem1);
    free(mem2);
  }
}
MAKE_CUDA_KERNEL(destroy_at);

// ------------------------------------------------------------
// construct_at
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(construct_at)
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

  S * ptr = um2::construct_at(reinterpret_cast<S *>(storage), 42, 2.71828F, 3.1415);
  ASSERT((*ptr).x == 42);
  ASSERT(((*ptr).y - 2.71828F) < 0.0001F);
  ASSERT(((*ptr).z - 3.1415) < 0.0001);
  um2::destroy_at(ptr);
}
MAKE_CUDA_KERNEL(construct_at);

// ------------------------------------------------------------
// destroy
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(destroy)
{
  {
    void * mem = malloc(5 * sizeof(Counted));
    ASSERT(mem != nullptr);
    ASSERT(count == 0);
    Counted * ptr_begin = nullptr;
    ptr_begin = ::new (mem) Counted();
    // Initialize the rest of the memory.
    for (size_t i = 1; i < 5; ++i) {
      void * mem_init =
          static_cast<void *>(static_cast<char *>(mem) + i * sizeof(Counted));
      ::new (mem_init) Counted();
    }
    ASSERT(ptr_begin != nullptr);
    Counted * ptr_end = ptr_begin + 5;
    ASSERT(count == 5);
    um2::destroy(ptr_begin + 2, ptr_end);
    ASSERT(count == 2);
    um2::destroy(ptr_begin, ptr_begin + 2);
    ASSERT(count == 0);
    free(mem);
  }
}
MAKE_CUDA_KERNEL(destroy);

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

TEST_SUITE(construct_at_suite)
{
  TEST_HOSTDEV(destroy_at);
  TEST_HOSTDEV(construct_at);
  TEST_HOSTDEV(destroy);
}

auto
main() -> int
{
  RUN_SUITE(construct_at_suite);
  return 0;
}
