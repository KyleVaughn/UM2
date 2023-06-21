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
TEST_CASE(test_basic_allocator)
{
  int32_t * ptr = nullptr;
  Foo * foo_ptr = nullptr;
  Size n = 1;
  um2::BasicAllocator<int32_t> alloc_i32;
  um2::BasicAllocator<Foo> alloc_foo;

  // max_size
  Size const size_max = static_cast<Size>(-1);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(alloc_i32.max_size() == size_max / 4);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(alloc_foo.max_size() == size_max / sizeof(Foo));

  // allocate
  ptr = alloc_i32.allocate(n);
  assert(ptr != nullptr);
  foo_ptr = alloc_foo.allocate(n);
  assert(foo_ptr != nullptr);

  // deallocate
  alloc_i32.deallocate(ptr, n);
  alloc_foo.deallocate(foo_ptr, n);

  // allocate_at_least
  n = 3;
  auto alloc_result_i32 = alloc_i32.allocate_at_least(n);
  assert(alloc_result_i32.ptr != nullptr);
  assert(alloc_result_i32.count == 3);
  alloc_i32.deallocate(alloc_result_i32.ptr, alloc_result_i32.count);
}
MAKE_CUDA_KERNEL(test_basic_allocator);

HOSTDEV
TEST_CASE(test_basic_allocator_traits)
{
  int32_t * ptr = nullptr;
  Foo * foo_ptr = nullptr;
  Size n = 1;
  um2::BasicAllocator<int32_t> alloc_i32;
  um2::BasicAllocator<Foo> alloc_foo;
  using AllocTraitsi32 = um2::AllocatorTraits<um2::BasicAllocator<int32_t>>;
  using AllocTraitsFoo = um2::AllocatorTraits<um2::BasicAllocator<Foo>>;

  // max_size
  Size const size_max = static_cast<Size>(-1);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(AllocTraitsi32::max_size(alloc_i32) == size_max / 4);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(AllocTraitsFoo::max_size(alloc_foo) == size_max / sizeof(Foo));

  // allocate
  ptr = AllocTraitsi32::allocate(alloc_i32, n);
  assert(ptr != nullptr);
  foo_ptr = AllocTraitsFoo::allocate(alloc_foo, n);
  assert(foo_ptr != nullptr);

  // deallocate
  AllocTraitsi32::deallocate(alloc_i32, ptr, n);
  AllocTraitsFoo::deallocate(alloc_foo, foo_ptr, n);

  // allocate_at_least
  n = 3;
  auto alloc_result_i32 = AllocTraitsi32::allocate_at_least(alloc_i32, n);
  assert(alloc_result_i32.ptr != nullptr);
  assert(alloc_result_i32.count == 3);
  AllocTraitsi32::deallocate(alloc_i32, alloc_result_i32.ptr, alloc_result_i32.count);
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

auto
main() -> int
{
  RUN_TESTS(basic_allocator);
  return 0;
}
