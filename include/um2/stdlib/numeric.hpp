#pragma once

#include <um2/config.hpp>

#include <numeric>

//==============================================================================
// NUMERIC
//==============================================================================
// Implementation of a subset of <numeric> which is compatible with CUDA.
// The following functions are implemented:
//  iota

namespace um2
{

//==============================================================================
// iota
//==============================================================================

template <class T>
HOSTDEV constexpr void
iota(T * first, T const * const last, T value) noexcept
{
  while (first != last) {
    *first++ = value;
    ++value;
  }
}

} // namespace um2
