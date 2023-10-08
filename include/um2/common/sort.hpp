#pragma once

#include <um2/stdlib/utility.hpp> // um2::move

namespace um2
{

//==============================================================================
// insertionSort
//==============================================================================

template <typename T>
HOSTDEV constexpr void
insertionSort(T * const first, T const * const last)
{
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
