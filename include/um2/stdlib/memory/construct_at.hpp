#pragma once

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/utility/forward.hpp>
#include <um2/stdlib/memory/voidify.hpp>

#include <new>

namespace um2
{

//==============================================================================
// construct_at
//==============================================================================

template <class T, class... Args>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::construct_at
construct_at(T * location, Args &&... args) noexcept -> T *
{
  ASSERT_ASSUME(location != nullptr);
  return ::new (um2::voidify(*location)) T(um2::forward<Args>(args)...);
}

//==============================================================================
// destroy_at
//==============================================================================

template <class T>
HOSTDEV constexpr void
// NOLINTNEXTLINE(readability-identifier-naming) match std::destroy_at
destroy_at(T * loc) noexcept
{
  ASSERT_ASSUME(loc != nullptr);
  if constexpr (std::is_array_v<T>) {
    for (auto & elem : *loc) {
      destroy_at(um2::addressof(elem));
    }
  } else {
    loc->~T();
  }
}

//==============================================================================
// destroy
//==============================================================================

template <class ForwardIt>
HOSTDEV constexpr auto 
destroy(ForwardIt first, ForwardIt last) noexcept -> ForwardIt
{
  for (; first != last; ++first) {
    um2::destroy_at(um2::addressof(*first));
  }
  return first;
}

} // namespace um2
