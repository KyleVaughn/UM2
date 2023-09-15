#pragma once

#include <um2/config.hpp>

#include <type_traits>
#include <utility>

// Contains:
//  forward
//  move
//  swap

namespace um2
{

//==============================================================================---
// forward
//==============================================================================---
//
// https://en.cppreference.com/w/cpp/utility/forward

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
forward(std::remove_reference_t<T> & t) noexcept -> T &&
{
  // Forwards lvalues as either lvalues or as rvalues, depending on T.
  return static_cast<T &&>(t);
}

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
forward(std::remove_reference_t<T> && t) noexcept -> T &&
{
  // Forwards rvalues as rvalues and prohibits forwarding of lvalues.
  static_assert(!std::is_lvalue_reference_v<T>,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<T &&>(t);
}

//==============================================================================---
// move
//==============================================================================---
//
// https://en.cppreference.com/w/cpp/utility/move

template <class T>
HOSTDEV constexpr auto
move(T && t) noexcept -> std::remove_reference_t<T> &&
{
  return static_cast<std::remove_reference_t<T> &&>(t);
}

//==============================================================================---
// swap
//==============================================================================---
//
// https://en.cppreference.com/w/cpp/utility/swap

#ifdef __CUDA_ARCH__

template <class T>
  requires(std::is_trivially_move_constructible_v<T> &&
           std::is_trivially_move_assignable_v<T>)
DEVICE constexpr void
swap(T & a, T & b) noexcept
{
  T tmp = um2::move(a);
  a = um2::move(b);
  b = um2::move(tmp);
}

#else

using std::swap;

#endif

} // namespace um2
