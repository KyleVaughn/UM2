#pragma once

#include <um2/config.hpp>

#include <um2/common/memory/addressof.hpp>
#include <um2/common/utility/forward.hpp>

#include <new>

namespace um2
{

// -----------------------------------------------------------------------------
// constructAt
// -----------------------------------------------------------------------------
// Constructs an object of type T in allocated uninitialized storage pointed to
// by p.
//
// https://en.cppreference.com/w/cpp/memory/construct_at

template <class T, class... Args>
HOSTDEV constexpr auto
constructAt(T * p, Args &&... args) -> T *
{
  assert(p != nullptr && "null pointer given to construct_at");
  return ::new (static_cast<void *>(p)) T(forward<Args>(args)...);
}

// -----------------------------------------------------------------------------
// destroyAt
// -----------------------------------------------------------------------------
// If T is not an array type, calls the destructor of the object pointed to by p,
// as if by p->~T().
// If T is an array type, the program recursively destroys elements of *p in order,
// as if by calling std::destroy(std::begin(*p), std::end(*p)).
//
// https://en.cppreference.com/w/cpp/memory/destroy_at

template <class T>
HOSTDEV constexpr void
destroyAt(T * p)
{
  if constexpr (std::is_array_v<T>) {
    for (auto & elem : *p) {
      destroyAt(addressof(elem));
    }
  } else {
    p->~T();
  }
}

// -----------------------------------------------------------------------------
// destroy
// -----------------------------------------------------------------------------
// Destroys the objects in the range [first, last).
//
// https://en.cppreference.com/w/cpp/memory/destroy

template <class ForwardIt>
HOSTDEV constexpr void
destroy(ForwardIt first, ForwardIt last)
{
  for (; first != last; ++first) {
    destroyAt(addressof(*first));
  }
}

} // namespace um2
