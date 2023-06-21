#pragma once

#include <um2/config.hpp>

#include <um2/common/memory/construct_at.hpp>

#include <type_traits>

// NOLINTBEGIN(readability-identifier-naming)
namespace um2
{

template <class Pointer>
struct AllocationResult {
  Pointer ptr;
  Size count; 
};

template <class Allocator>
struct AllocatorTraits {

  using value_type = typename Allocator::value_type;
  using pointer = value_type *;

  HOSTDEV [[nodiscard]] constexpr static auto
  allocate(Allocator & a, Size n) noexcept -> pointer
  {
    return a.allocate(n);
  }

  HOSTDEV constexpr static void
  deallocate(Allocator & a, pointer p, Size n) noexcept
  {
    a.deallocate(p, n);
  }

  HOSTDEV constexpr static void
  destroy(Allocator & /*a*/, pointer p) noexcept
  {
    destroy_at(p);
  }

  HOSTDEV [[nodiscard]] constexpr static auto
  max_size(Allocator const & a) noexcept -> Size
  {
    return a.max_size();
  }

  HOSTDEV [[nodiscard]] constexpr static auto
  allocate_at_least(Allocator & a, Size n) noexcept -> AllocationResult<pointer>
  {
    if constexpr (requires { a.allocate_at_least(n); }) {
      return a.allocate_at_least(n);
    } else {
      return {a.allocate(n), n};
    }
  }

}; // struct AllocatorTraits

} // namespace um2
// NOLINTEND(readability-identifier-naming)
