#pragma once

#include <um2/config.hpp>

#include <memory>

namespace um2
{

// -----------------------------------------------------------------------------
// addressof
// -----------------------------------------------------------------------------
// Obtains the actual address of the object or function arg, even in presence of
// overloaded operator&
// https://en.cppreference.com/w/cpp/memory/addressof

#ifndef __CUDA_ARCH__

template <class T>
constexpr auto
addressof(T & arg) noexcept -> T *
{
  return std::addressof(arg);
}

#else

template <class T>
__device__ constexpr auto
addressof(T & arg) noexcept -> T *
{
  return __builtin_addressof(arg);
}

#endif

} // namespace um2
