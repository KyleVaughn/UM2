#pragma once

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/memory/voidify.hpp>
#include <um2/stdlib/utility/forward.hpp>

#include <new>

namespace um2
{

//==============================================================================
// construct_at
//==============================================================================
// This cannot be constexpr because it uses placement new. In the C++ stdlib
// their construct_at IS constexpr because they have cheat codes... :(

template <class T, class... Args>
HOSTDEV inline auto
// match std::construct_at
// NOLINTNEXTLINE(readability-identifier-naming,*missing-std-forward) OK
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
  static_assert(!std::is_array_v<T>, "destroy_at does not support arrays");
  loc->~T();
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
