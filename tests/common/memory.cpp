#include "../test_framework.hpp"
#include <um2/common/memory.hpp>

struct Foo
{
  int32_t a;
  int32_t b;
  int32_t c;
  int32_t d;
};

struct A {
  void operator&() const {}
};

struct Nothing {
    operator char&() {
        static char c;
        return c;
    }
};

#ifndef __CUDA_ARCH__
int count = 0;
#else
__device__ int count = 0;
#endif
struct Counted {
  HOSTDEV static void reset() { count = 0; }
  HOSTDEV Counted() { ++count; }
  HOSTDEV Counted(Counted const&) { ++count; }
  HOSTDEV ~Counted() { --count; }
  HOSTDEV friend void operator&(Counted) = delete;
};

struct VCounted {
  HOSTDEV static void reset() { count = 0; }
  HOSTDEV VCounted() { ++count; }
  HOSTDEV VCounted(VCounted const&) { ++count; }
  HOSTDEV virtual ~VCounted() { --count; }
  HOSTDEV friend void operator&(VCounted) = delete;
};

struct DCounted : VCounted {
  HOSTDEV friend void operator&(DCounted) = delete;
};

// ------------------------------------------------------------
// addressof
// ------------------------------------------------------------

HOSTDEV TEST_CASE(addressof)
{
  {
    int i;
    double d;
    EXPECT_EQ(um2::addressof(i), &i);
    EXPECT_EQ(um2::addressof(d), &d);

    A* tp = new A;
    const A* ctp = tp;
    EXPECT_EQ(um2::addressof(*tp), tp);
    EXPECT_EQ(um2::addressof(*ctp), ctp);
    delete tp;
  }

  {
    union {
      Nothing n;
      int i;
    };
    EXPECT_EQ(um2::addressof(n), static_cast<void*>(um2::addressof(n)))
  }

}
MAKE_CUDA_KERNEL(addressof);

// ------------------------------------------------------------
// destroy_at
// ------------------------------------------------------------

HOSTDEV TEST_CASE(destroy_at)
{
  {
    void* mem1 = malloc(sizeof(Counted));
    void* mem2 = malloc(sizeof(Counted));
    EXPECT_NE(mem1, nullptr);
    EXPECT_NE(mem2, nullptr);
    EXPECT_EQ(count, 0);
    Counted* ptr1 = ::new(mem1) Counted();
    Counted* ptr2 = ::new(mem2) Counted();
    EXPECT_EQ(count, 2);
    um2::destroy_at(ptr1);
    EXPECT_EQ(count, 1);
    um2::destroy_at(ptr2);
    EXPECT_EQ(count, 0);
    free(mem1);
    free(mem2);
    count = 0;
  }
  {
    void* mem1 = malloc(sizeof(DCounted));
    void* mem2 = malloc(sizeof(DCounted));
    EXPECT_NE(mem1, nullptr);
    EXPECT_NE(mem2, nullptr);
    EXPECT_EQ(count, 0);
    DCounted* ptr1 = ::new(mem1) DCounted();
    DCounted* ptr2 = ::new(mem2) DCounted();
    EXPECT_EQ(count, 2);
    um2::destroy_at(ptr1);
    EXPECT_EQ(count, 1);
    um2::destroy_at(ptr2);
    EXPECT_EQ(count, 0);
    free(mem1);
    free(mem2);
  }
}
MAKE_CUDA_KERNEL(destroy_at);

// ------------------------------------------------------------
// allocator
// ------------------------------------------------------------

HOSTDEV TEST_CASE(allocator)
{
  int32_t* ptr = nullptr;
  um2::Allocator<int32_t> alloc_i32;
  ptr = alloc_i32.allocate(1);
  EXPECT_NE(ptr, nullptr);
  alloc_i32.deallocate(ptr);

  Foo* foo_ptr = nullptr;
  um2::Allocator<Foo> alloc_foo;
  foo_ptr = alloc_foo.allocate(1);
  EXPECT_NE(foo_ptr, nullptr);
  alloc_foo.deallocate(foo_ptr);
}
MAKE_CUDA_KERNEL(allocator);

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

TEST_SUITE(memory)
{
  TEST_HOSTDEV(addressof);
  TEST_HOSTDEV(destroy_at);
  TEST_HOSTDEV(allocator);
}

auto main() -> int
{
  RUN_TESTS(memory);
  return 0;
}
