#pragma once

#include <um2/config.hpp>

#include <type_traits>

// NOLINTBEGIN(readability-identifier-naming)
namespace um2
{

template <class Allocator>
struct AllocatorTraits {

  using value_type = typename Allocator::value_type;
  using pointer = value_type *; 

  HOSTDEV [[nodiscard]] constexpr static auto 
  allocate(Allocator & a, size_t n) noexcept -> pointer {
      return a.allocate(n);
  }

  HOSTDEV constexpr static 
  void deallocate(Allocator & a, pointer p, size_t n) noexcept {
      a.deallocate(p, n);
  }

  HOSTDEV [[nodiscard]] constexpr static auto 
  max_size(Allocator const & a) noexcept -> size_t { 
    return a.max_size();
  }

}; // struct AllocatorTraits

} // namespace um2
// NOLINTEND(readability-identifier-naming)
