#pragma once

#include <um2/stdlib/algorithm.hpp> // std::sort
#include <um2/stdlib/numeric.hpp>   // std::iota
#include <um2/stdlib/utility.hpp>   // um2::move
#include <um2/stdlib/vector.hpp>    // um2::Vector

//==============================================================================
// SORT
//==============================================================================
// This file contains sorting algorithms and related functions.
//
// The following functions are provided:
// insertionSort
// sortPermutation
// applyPermutation
// invertPermutation

namespace um2
{

//==============================================================================
// insertionSort
//==============================================================================
// Use the insertion sort algorithm to sort [first, last) in-place.
// This should be used for small arrays (size < 20) or when the array is
// already mostly sorted.

template <typename T>
HOSTDEV constexpr void
insertionSort(T * const first, T const * const last) noexcept
{
  // Not the clearest implementation, but the assembly is much better than
  // the obvious implementation found on Wikipedia.
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

//==============================================================================
// sortPermutation
//==============================================================================
// Create a permutation that sorts [begin, end) when applied. [begin, end) is
// not modified.

template <typename T>
constexpr void
sortPermutation(T const * const begin, T const * const end,
                Size * const perm_begin) noexcept
{
  auto const n = end - begin;
  std::iota(perm_begin, perm_begin + n, 0);
  std::sort(perm_begin, perm_begin + n,
            [&begin](Size const i, Size const j) { return begin[i] < begin[j]; });
}

//==============================================================================
// applyPermutation
//==============================================================================
// Apply the permutation perm to the vector v in-place.

template <typename T>
constexpr void
applyPermutation(Vector<T> & v, Vector<Size> const & perm) noexcept
{
  // Verify that perm is a permutation
  // (contains all elements of [0, v.size()) exactly once)
#if UM2_ENABLE_ASSERTS
  Vector<int8_t> seen(v.size(), 0);
  for (Size const i : perm) {
    ASSERT(i < v.size());
    ASSERT(seen[i] == 0);
    seen[i] = 1;
  }
  for (int8_t const i : seen) {
    ASSERT(i == 1);
  }
#endif
  // Apply the permutation in-place by iterating over cycles.
  Vector<int8_t> done(v.size(), 0);
  for (Size i = 0; i < v.size(); ++i) {
    if (done[i] == 1) {
      continue;
    }
    done[i] = 1;
    Size prev_j = i;
    Size j = perm[i];
    while (i != j) {
      um2::swap(v[prev_j], v[j]);
      done[j] = 1;
      prev_j = j;
      j = perm[j];
    }
  }
}

//==============================================================================
// invertPermutation
//==============================================================================
// Compute the inverse of the permutation perm and store it in inv_perm.

template <std::integral I>
void
invertPermutation(Vector<I> const & perm, Vector<I> & inv_perm) noexcept
{
  ASSERT(perm.size() == inv_perm.size());
  for (Size i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }
}

} // namespace um2
