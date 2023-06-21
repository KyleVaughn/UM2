#pragma once

#include <um2/common/memory/AllocatorTraits.hpp>

// NOLINTBEGIN(readability-identifier-naming)
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

  using value_type = T;

  HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionStatic
  max_size() const noexcept -> Size
  {
    return static_cast<Size>(-1) / static_cast<Size>(sizeof(T));
  }

  HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionStatic
  allocate(Size n) noexcept -> T *
  {
    return static_cast<T *>(::operator new(n * sizeof(T)));
  }

  HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionStatic
  allocate_at_least(Size n) noexcept -> AllocationResult<T *>
  {
    return {allocate(n), n};
  }

  HOSTDEV constexpr void
  // cppcheck-suppress functionStatic
  deallocate(T * p, Size /*n*/) noexcept
  {
    ::operator delete(p);
  }

}; // struct BasicAllocator

} // namespace um2
// NOLINTEND(readability-identifier-naming)
