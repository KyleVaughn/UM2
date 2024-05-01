#pragma once

#include <um2/config.hpp>

namespace um2
{

template <typename InputIt1, typename InputIt2>
HOSTDEV PURE constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::lexicographical_compare
lexicographical_compare(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
    -> bool
{
  for (; first2 != last2; ++first1, ++first2) {
    if (first1 == last1 || *first1 < *first2) {
      return true;
    }
    if (*first2 < *first1) {
      return false;
    }
  }
  return false;
}

} // namespace um2
