#pragma once

#include <um2/common/memory/AllocatorTraits.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// BASIC ALLOCATOR
// -----------------------------------------------------------------------------
// An std::allocator-like class.
//
// https://en.cppreference.com/w/cpp/memory/allocator

template <class T>
struct BasicAllocator {

  using Value = T;
  using Pointer = T *;

  HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionStatic
  maxSize() const noexcept -> Size
  {
    return static_cast<Size>(-1) / static_cast<Size>(sizeof(T));
  }

  HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionStatic
  allocate(Size n) noexcept -> Pointer
  {
    return static_cast<Pointer>(::operator new(n * sizeof(T)));
  }

  HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionStatic
  allocateAtLeast(Size n) noexcept -> AllocationResult<Pointer>
  {
    return {allocate(n), n};
  }

  HOSTDEV constexpr void
  // cppcheck-suppress functionStatic
  deallocate(Pointer p, Size /*n*/) noexcept
  {
    ::operator delete(p);
  }

}; // struct BasicAllocator

} // namespace um2
