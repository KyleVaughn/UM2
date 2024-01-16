#pragma once

#include <um2/stdlib/utility.hpp>

#include <new>

//==============================================================================
// MEMORY
//==============================================================================
// Implementation of a subset of <memory> which is compatible with CUDA.
// See https://en.cppreference.com/w/cpp/header/memory for details.
// The following functions are implemented:
//  addressof
//  construct_at
//  destroy_at
//  destroy

namespace um2
{

//==============================================================================
// addressof
//==============================================================================
// Obtains the actual address of the object or function arg, even in presence of
// overloaded operator &.

template <class T>
HOSTDEV constexpr auto
addressof(T & arg) noexcept -> T *
{
  return __builtin_addressof(arg);
}

//==============================================================================
// construct_at
//==============================================================================
// Constructs an object of type T in allocated uninitialized storage pointed to
// by p.

template <class T, class... Args>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::construct_at
construct_at(T * p, Args &&... args) noexcept -> T *
{
  ASSERT_ASSUME(p != nullptr);
  return ::new (static_cast<void *>(p)) T(um2::forward<Args>(args)...);
}

//==============================================================================
// destroy_at
//==============================================================================
// If T is not an array type, calls the destructor of the object pointed to by p,
// as if by p->~T().
//
// If T is an array type, the program recursively destroys elements of *p in order,
// as if by calling std::destroy(std::begin(*p), std::end(*p)).

template <class T>
HOSTDEV constexpr void
// NOLINTNEXTLINE(readability-identifier-naming) match std::destroy_at
destroy_at(T * p) noexcept
{
  if constexpr (std::is_array_v<T>) {
    for (auto & elem : *p) {
      destroy_at(um2::addressof(elem));
    }
  } else {
    p->~T();
  }
}

//==============================================================================
// destroy
//==============================================================================
// Destroys the objects in the range [first, last).

template <class ForwardIt>
HOSTDEV constexpr void
destroy(ForwardIt first, ForwardIt last) noexcept
{
  for (; first != last; ++first) {
    um2::destroy_at(um2::addressof(*first));
  }
}

} // namespace um2
