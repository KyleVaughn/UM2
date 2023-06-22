#pragma once

#include <um2/config.hpp>

#include <um2/common/memory/constructAt.hpp>

namespace um2
{

template <class Pointer>
struct AllocationResult {
  Pointer ptr;
  Size count;
};

template <class Allocator>
struct AllocatorTraits {

  using Value = typename Allocator::Value;
  using Pointer = Value *;

  HOSTDEV [[nodiscard]] constexpr static auto
  allocate(Allocator & a, Size n) noexcept -> Pointer
  {
    return a.allocate(n);
  }

  HOSTDEV constexpr static void
  deallocate(Allocator & a, Pointer p, Size n) noexcept
  {
    a.deallocate(p, n);
  }

  HOSTDEV constexpr static void
  destroy(Allocator & a, Pointer p) noexcept
  {
    if constexpr (requires { a.destroy(p); }) {
      a.destroy(p);
    } else {
      destroyAt(p);
    }
  }

  template <class... Args>
  HOSTDEV constexpr static void
  construct(Allocator & a, Pointer p, Args &&... args) noexcept
  {
    if constexpr (requires { a.construct(p, forward<Args>(args)...); }) {
      a.construct(p, forward<Args>(args)...);
    } else {
      constructAt(p, forward<Args>(args)...);
    }
  }

  HOSTDEV [[nodiscard]] constexpr static auto
  maxSize(Allocator const & a) noexcept -> Size
  {
    if constexpr (requires { a.maxSize(); }) {
      return a.maxSize();
    } else {
      return static_cast<Size>(-1) / sizeof(Value);
    }
  }

  HOSTDEV [[nodiscard]] constexpr static auto
  allocateAtLeast(Allocator & a, Size n) noexcept -> AllocationResult<Pointer>
  {
    if constexpr (requires { a.allocateAtLeast(n); }) {
      return a.allocateAtLeast(n);
    } else {
      return {a.allocate(n), n};
    }
  }

}; // struct AllocatorTraits

} // namespace um2
