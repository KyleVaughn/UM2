#pragma once

#include <um2/config.hpp>

namespace um2
{

//==============================================================================
// addressof
//==============================================================================
//
// Obtains the actual address of the object or function arg, even in presence of
// overloaded operator &.
//
// https://en.cppreference.com/w/cpp/memory/addressof

template <class T>
HOSTDEV constexpr auto
addressof(T & arg) noexcept -> T *
{
  return __builtin_addressof(arg);
}

} // namespace um2
