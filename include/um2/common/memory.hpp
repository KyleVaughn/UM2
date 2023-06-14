#pragma once

#include <um2/config.hpp>

#include <memory>

namespace um2
{

// -----------------------------------------------------------------------------
// addressof 
// -----------------------------------------------------------------------------
// Returns the actual address of the object or function arg, even in presence of 
// overloaded operator&.
// https://en.cppreference.com/w/cpp/memory/addressof

#ifndef __CUDA_ARCH__

template <typename T>
constexpr auto addressof(T& arg) noexcept -> T*
{
  return std::addressof(arg);
}

#else

template <typename T>
requires (std::is_object_v<T>)
__device__ constexpr auto addressof(T& arg) noexcept -> T*
{
  return reinterpret_cast<T*>(
             &const_cast<char&>(
                 reinterpret_cast<const volatile char&>(arg)));
}

template <typename T>
requires (!std::is_object_v<T>)
__device__ constexpr auto addressof(T& arg) noexcept -> T*
{
  return &arg;
}

template <typename T>
const T* addressof(const T&&) = delete;

#endif

// -----------------------------------------------------------------------------
// destroy_at
// -----------------------------------------------------------------------------
// If T is not an array type, calls the destructor of the object pointed to by p, 
// as if by p->~T().
// If T is an array type, the program recursively destroys elements of *p in order, 
// as if by calling std::destroy(std::begin(*p), std::end(*p)). 
// https://en.cppreference.com/w/cpp/memory/destroy_at

#ifndef __CUDA_ARCH__

template <typename T>
constexpr void destroy_at(T* p)
{
  std::destroy_at(p); 
}

#else

template <typename T>
__device__ constexpr void destroy_at(T* p)
{
  if constexpr (std::is_array_v<T>) {
    for (auto& elem : *p) {
      destroy_at(addressof(elem));
    }
  } else {
    p->~T();
  } 
}

#endif

// -----------------------------------------------------------------------------
// ALLOCATOR 
// -----------------------------------------------------------------------------
// An std::allocator-like class.
//
// https://en.cppreference.com/w/cpp/memory/allocator

template <typename T>
struct Allocator {

  HOSTDEV [[nodiscard]] constexpr auto allocate(size_t n) -> T*
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  // We want the Allocator to have a common interface with std::allocator, so we
  // don't want to make this function static.
  // cppcheck-suppress functionStatic
  HOSTDEV constexpr void deallocate(T * p)
  {
    ::operator delete(p);
  }

}; // struct Allocator 

} // namespace um2
