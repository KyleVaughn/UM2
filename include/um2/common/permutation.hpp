#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/numeric.hpp>

namespace um2
{

//==============================================================================
// Sorting
//==============================================================================

template <typename T>
constexpr void
sortPermutation(T const * const begin,
                T const * const end,
                Size * const perm_begin) noexcept
{
  auto const n = end - begin;
  std::iota(perm_begin, perm_begin + n, 0);
  std::sort(perm_begin, perm_begin + n,
            [&begin](Size const i, Size const j) { return begin[i] < begin[j]; });
}

//template <typename T>    
//constexpr void    
//applyPermutation(Vector<T> & v, Vector<Size> const & perm) noexcept    
//{    
//  // Verify that perm is a permutation    
//  // (contains all elements of [0, v.size()) exactly once)    
//#ifndef NDEBUG    
//  Vector<int8_t> seen(v.size(), 0);    
//  for (Size const i : perm) {    
//    assert(i < v.size());    
//    assert(seen[i] == 0);    
//    seen[i] = 1;    
//  }    
//  assert(std::count(seen.cbegin(), seen.cend(), 1) == v.size());    
//#endif    
//  // Apply the permutation in-place by iterating over cycles.    
//  Vector<int8_t> done(v.size());    
//  for (Size i = 0; i < v.size(); ++i) {    
//    if (done[i] == 1) {    
//      continue;    
//    }    
//    done[i] = true;    
//    Size prev_j = i;    
//    Size j = perm[i];    
//    while (i != j) {    
//      std::swap(v[prev_j], v[j]);    
//      done[j] = true;    
//      prev_j = j;    
//      j = perm[j];    
//    }    
//  }    
//}

} // namespace um2
