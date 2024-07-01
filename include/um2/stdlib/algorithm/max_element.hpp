#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class ForwardIt>
PURE HOSTDEV [[nodiscard]] constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::max_element
max_element(ForwardIt first, ForwardIt last) noexcept -> ForwardIt
{
  if (first != last) {
    ForwardIt i = first;
    while (++i != last) {
      if (*first < *i) {
        first = i;
      }
    }
  }
  return first;
}

} // namespace um2
