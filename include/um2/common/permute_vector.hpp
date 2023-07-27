#pragma once

#include <numeric>
#include <vector>

namespace um2
{

template <typename T>
constexpr void
sortPermutation(std::vector<T> const & v, std::vector<size_t> & perm) noexcept
{
  perm.resize(v.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&v](size_t const i, size_t const j) { return v[i] < v[j]; });
}

template <typename T>
constexpr void
applyPermutation(std::vector<T> & v, std::vector<size_t> const & perm) noexcept
{
  // Verify that perm is a permutation
  // (contains all elements of [0, v.size()) exactly once)
#ifndef NDEBUG
  std::vector<int8_t> seen(v.size(), 0);
  for (size_t const i : perm) {
    assert(i < v.size());
    assert(seen[i] == 0);
    seen[i] = 1;
  }
  assert(std::count(seen.cbegin(), seen.cend(), 1) == static_cast<long>(v.size()));
#endif
  // Apply the permutation in-place by iterating over cycles.
  std::vector<int8_t> done(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    if (done[i] == 1) {
      continue;
    }
    done[i] = true;
    size_t prev_j = i;
    size_t j = perm[i];
    while (i != j) {
      std::swap(v[prev_j], v[j]);
      done[j] = true;
      prev_j = j;
      j = perm[j];
    }
  }
}

} // namespace um2
