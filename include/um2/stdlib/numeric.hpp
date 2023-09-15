#pragma once

#include <um2/config.hpp>

#include <numeric>

namespace um2
{

//==============================================================================
// iota
//==============================================================================

template <class T>
HOSTDEV constexpr void
iota(T * first, T const * const last, T value)
{
  while (first != last) {
    *first++ = value;
    ++value;
  }
}

} // namespace um2
