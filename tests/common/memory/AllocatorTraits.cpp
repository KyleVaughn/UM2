#include "../../test_macros.hpp"
#include <um2/common/memory/BasicAllocator.hpp>

struct Foo {
  int32_t a;
  int32_t b;
  int32_t c;
  int32_t d;
};

HOSTDEV
TEST_CASE(test_AllocatorTraits)
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
  assert(AllocTraitsi32::maxSize(alloc_i32) == size_max / 4);
  // NOLINTNEXTLINE(misc-static-assert)
  assert(AllocTraitsFoo::maxSize(alloc_foo) == size_max / sizeof(Foo));

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
  auto alloc_result_i32 = AllocTraitsi32::allocateAtLeast(alloc_i32, n);
  assert(alloc_result_i32.ptr != nullptr);
  assert(alloc_result_i32.count == 3);
  AllocTraitsi32::deallocate(alloc_i32, alloc_result_i32.ptr, alloc_result_i32.count);
}
MAKE_CUDA_KERNEL(test_AllocatorTraits);

// ------------------------------------------------------------
// Test Suite
// ------------------------------------------------------------

TEST_SUITE(allocator_traits) { TEST_HOSTDEV(test_AllocatorTraits); }

auto
main() -> int
{
  RUN_TESTS(allocator_traits);
  return 0;
}
