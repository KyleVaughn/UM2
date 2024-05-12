#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class InputIt1, class InputIt2>
PURE HOSTDEV [[nodiscard]] constexpr auto
equal(InputIt1 first1, InputIt1 last1, InputIt2 first2) noexcept -> bool
{
  for (; first1 != last1; ++first1, ++first2) {
    if (!(*first1 == *first2)) {
      return false;
    }
  }
  return true;
}

} // namespace um2
