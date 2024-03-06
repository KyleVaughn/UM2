#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/move.hpp>

namespace um2
{

//==============================================================================
// insertionSort
//==============================================================================
// Use the insertion sort algorithm to sort [first, last) in-place.
// This should be used for small arrays (size < 20) or when the array is
// already mostly sorted.

template <typename T>
HOSTDEV void
insertionSort(T * const first, T const * const last) noexcept
{
  // Not the clearest implementation, but the assembly is much better than
  // the obvious implementation
  if (first == last) {
    return;
  }
  T * i = first + 1;
  for (; i != last; ++i) {
    T * j = i - 1;
    if (*i < *j) {
      T t = um2::move(*i);
      T * k = j;
      j = i;
      do {
        *j = um2::move(*k);
        j = k;
      } while (j != first && t < *--k);
      *j = um2::move(t);
    }
  }
}

} // namespace um2
