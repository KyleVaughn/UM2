#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/swap.hpp>

namespace um2
{

//==============================================================================
// sort3
//==============================================================================

template <class T>
HOSTDEV constexpr void
sort3(T * const x, T * const y, T * const z) noexcept
{
  if (!(*y < *x)) {   // if x <= y
    if (!(*z < *y)) { // if y <= z
      return;         // x <= y && y <= z
    }
    // x <= y && y > z
    um2::swap(*y, *z);   // x <= z && y < z
    if (*y < *x) {       // if x > y
      um2::swap(*x, *y); // x < y && y <= z
    }
    return; // x <= y && y < z
  }

  // x > y
  if (*z < *y) {       // if y > z
    um2::swap(*x, *z); // x < y && y < z
    return;
  }
  // x > y && y <= z
  um2::swap(*x, *y); // x > y && y <= z
  // x < y && x <= z
  if (*z < *y) {       // if y > z
    um2::swap(*y, *z); // x <= y && y < z
  }
}

//==============================================================================
// sort4
//==============================================================================

template <class T>
HOSTDEV constexpr void
sort4(T * const x1, T * const x2, T * const x3, T * const x4) noexcept
{
  um2::sort3(x1, x2, x3);
  if (*x4 < *x3) {
    um2::swap(*x3, *x4);
    if (*x3 < *x2) {
      um2::swap(*x2, *x3);
      if (*x2 < *x1) {
        um2::swap(*x1, *x2);
      }
    }
  }
}

} // namespace um2
