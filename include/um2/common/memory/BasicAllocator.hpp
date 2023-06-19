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

template <typename T>
struct BasicAllocator {

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type = T;

  //cppcheck-suppress functionStatic
  HOSTDEV [[nodiscard]] constexpr auto max_size() const noexcept -> size_t
  {
    return static_cast<size_t>(-1) / sizeof(T);
  }

  //cppcheck-suppress functionStatic
  HOSTDEV [[nodiscard]] constexpr auto allocate(size_t n) noexcept -> T*
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  // cppcheck-suppress functionStatic
  HOSTDEV constexpr void deallocate(T * p, size_t /*n*/) noexcept
  {
    ::operator delete(p);
  }

  // NOLINTEND(readability-identifier-naming)
}; // struct BasicAllocator 

} // namespace um2
