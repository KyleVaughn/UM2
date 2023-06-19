#include "../../test_macros.hpp"
#include <um2/common/memory/BasicAllocator.hpp>

struct Foo
{
  int32_t a;
  int32_t b;
  int32_t c;
  int32_t d;
};

// ------------------------------------------------------------
// BasicAllocator 
// ------------------------------------------------------------

HOSTDEV TEST_CASE(test_basic_allocator)
{
  int32_t* ptr = nullptr;
  Foo* foo_ptr = nullptr;
  size_t n = 1;
  um2::BasicAllocator<int32_t> alloc_i32;
  um2::BasicAllocator<Foo> alloc_foo;

  // allocate
  ptr = alloc_i32.allocate(n);
  assert(ptr != nullptr);
  foo_ptr = alloc_foo.allocate(n);
  assert(foo_ptr != nullptr);

  // deallocate
  alloc_i32.deallocate(ptr, n);
  alloc_foo.deallocate(foo_ptr, n);
}
MAKE_CUDA_KERNEL(test_basic_allocator);

HOSTDEV TEST_CASE(test_basic_allocator_traits)
{
  int32_t* ptr = nullptr;
  Foo* foo_ptr = nullptr;
  size_t n = 1;
  um2::BasicAllocator<int32_t> alloc_i32;
  um2::BasicAllocator<Foo> alloc_foo;

  // max_size
  size_t const size_max = ~0ULL;
  // NOLINTNEXTLINE(misc-static-assert)
  assert(um2::AllocatorTraits<um2::BasicAllocator<int32_t>>::max_size(alloc_i32) == size_max/4);

  // allocate
  ptr = um2::AllocatorTraits<um2::BasicAllocator<int32_t>>::allocate(alloc_i32, n); 
  assert(ptr != nullptr);
  foo_ptr = um2::AllocatorTraits<um2::BasicAllocator<Foo>>::allocate(alloc_foo, n); 
  assert(foo_ptr != nullptr);

  // deallocate
  um2::AllocatorTraits<um2::BasicAllocator<int32_t>>::deallocate(alloc_i32, ptr, n); 
  um2::AllocatorTraits<um2::BasicAllocator<Foo>>::deallocate(alloc_foo, foo_ptr, n); 
}
MAKE_CUDA_KERNEL(test_basic_allocator_traits);

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

TEST_SUITE(basic_allocator)
{
  TEST_HOSTDEV(test_basic_allocator);
  TEST_HOSTDEV(test_basic_allocator_traits);
}

auto main() -> int
{
  RUN_TESTS(basic_allocator);
  return 0;
}
