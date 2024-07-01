#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class ForwardIt>
PURE HOSTDEV [[nodiscard]] constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::is_sorted
is_sorted(ForwardIt first, ForwardIt last) noexcept -> bool
{
  if (first != last) {
    ForwardIt i = first;
    while (++i != last) {
      if (*i < *first) {
        return false;
      }
      first = i;
    }
  }
  return true;
}

} // namespace um2
