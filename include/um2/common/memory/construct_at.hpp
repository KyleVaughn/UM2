#pragma once

#include <um2/config.hpp>

#include <um2/common/memory/addressof.hpp>

#include <memory>

// NOLINTBEGIN(readability-identifier-naming)
namespace um2
{

// -----------------------------------------------------------------------------
// destroy_at
// -----------------------------------------------------------------------------
// If T is not an array type, calls the destructor of the object pointed to by p,
// as if by p->~T().
// If T is an array type, the program recursively destroys elements of *p in order,
// as if by calling std::destroy(std::begin(*p), std::end(*p)).
// https://en.cppreference.com/w/cpp/memory/destroy_at

#ifndef __CUDA_ARCH__

template <class T>
constexpr void
destroy_at(T * p)
{
  std::destroy_at(p);
}

#else

template <class T>
__device__ constexpr void
destroy_at(T * p)
{
  if constexpr (std::is_array_v<T>) {
    for (auto & elem : *p) {
      destroy_at(addressof(elem));
    }
  } else {
    p->~T();
  }
}

#endif

// -----------------------------------------------------------------------------
// destroy
// -----------------------------------------------------------------------------
// Destroys the objects in the range [first, last).
// https://en.cppreference.com/w/cpp/memory/destroy

#ifndef __CUDA_ARCH__

template <class ForwardIt>
constexpr void
destroy(ForwardIt first, ForwardIt last)
{
  std::destroy(first, last);
}

#else

template <class ForwardIt>
__device__ constexpr void
destroy(ForwardIt first, ForwardIt last)
{
  for (; first != last; ++first) {
    destroy_at(addressof(*first));
  }
}

#endif

} // namespace um2
// NOLINTEND(readability-identifier-naming)
