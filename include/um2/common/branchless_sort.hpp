#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/swap.hpp>

namespace um2
{

//==============================================================================
// sort3 
//==============================================================================

template <typename T>
HOSTDEV constexpr void
sort3(T * const x, T * const y, T * const z) noexcept
{
  if (!(*y < *x)) { // if x <= y
    if (!(*z < *y)) { // if y <= z
      return; // x <= y && y <= z
    }
    // x <= y && y > z
    um2::swap(*y, *z); // x <= z && y < z
    if (*y < *x) { // if x > y
      um2::swap(*x, *y); // x < y && y <= z
    }
    return; // x <= y && y < z
  }
  
  // x > y
  if (*z < *y) { // if y > z
    um2::swap(*x, *z); // x < y && y < z
    return;
  }
  // x > y && y <= z
  um2::swap(*x, *y); // x > y && y <= z
  // x < y && x <= z
  if (*z < *y) { // if y > z
    um2::swap(*y, *z); // x <= y && y < z
  }
}

} // namespace um2
