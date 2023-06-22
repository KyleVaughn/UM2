#include "../../test_macros.hpp"
#include <um2/common/memory/BasicAllocator.hpp>

struct Foo {
  int32_t a;
  int32_t b;
  int32_t c;
  int32_t d;
};

// ------------------------------------------------------------
// BasicAllocator
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(test_BasicAllocator)
{
  int32_t * ptr = nullptr;
  Foo * foo_ptr = nullptr;
  Size n = 1;
  um2::BasicAllocator<int32_t> alloc_i32;
  um2::BasicAllocator<Foo> alloc_foo;

  // maxSize
  Size const size_max = static_cast<Size>(-1);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(alloc_i32.maxSize() == size_max / 4);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(alloc_foo.maxSize() == size_max / sizeof(Foo));

  // allocate
  ptr = alloc_i32.allocate(n);
  assert(ptr != nullptr);
  foo_ptr = alloc_foo.allocate(n);
  assert(foo_ptr != nullptr);

  // deallocate
  alloc_i32.deallocate(ptr, n);
  alloc_foo.deallocate(foo_ptr, n);

  // allocateAtLeast
  n = 3;
  auto alloc_result_i32 = alloc_i32.allocateAtLeast(n);
  assert(alloc_result_i32.ptr != nullptr);
  assert(alloc_result_i32.count == 3);
  alloc_i32.deallocate(alloc_result_i32.ptr, alloc_result_i32.count);
}
MAKE_CUDA_KERNEL(test_basic_allocator);

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

TEST_SUITE(basic_allocator) { TEST_HOSTDEV(test_BasicAllocator); }

auto
main() -> int
{
  RUN_TESTS(basic_allocator);
  return 0;
}
