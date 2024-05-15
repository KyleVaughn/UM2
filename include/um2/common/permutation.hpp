#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/numeric/iota.hpp>
#include <um2/stdlib/utility/swap.hpp>

#include <algorithm> // std::sort

namespace um2
{

//==============================================================================
// sortPermutation
//==============================================================================
// Create a permutation that sorts [begin, end) when applied. [begin, end) is
// not modified.

template <typename T>
void
sortPermutation(T const * const begin, T const * const end,
                Int * const perm_begin) noexcept
{
  auto const n = end - begin;
  um2::iota(perm_begin, perm_begin + n, 0);
  std::sort(perm_begin, perm_begin + n,
            [&begin](Int const i, Int const j) { return begin[i] < begin[j]; });
}

//==============================================================================
// applyPermutation
//==============================================================================
// Apply the permutation perm to [begin, end).

template <typename T>
void
applyPermutation(T * const begin, T * const end, Int const * const perm) noexcept
{
  auto const size = static_cast<Int>(end - begin);
  if (size == 0) {
    return;
  }
  for (Int i = 0; i < size - 1; ++i) {
    Int ind = perm[i];
    ASSERT(0 <= ind);
    while (ind < i) {
      ASSERT(0 <= ind);
      ASSERT(ind < size);
      ind = perm[ind];
    }
    um2::swap(begin[i], begin[ind]);
  }
}

//==============================================================================
// invertPermutation
//==============================================================================
// Compute the inverse of the permutation perm and store it in inv_perm.

inline void
invertPermutation(Int const * const p_begin, Int const * const p_end,
                  Int * const inv_perm) noexcept
{
  auto const size = static_cast<Int>(p_end - p_begin);
  for (Int i = 0; i < size; ++i) {
    inv_perm[p_begin[i]] = i;
  }
}

} // namespace um2
