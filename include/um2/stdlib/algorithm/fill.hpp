#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class It, class T>
HOSTDEV inline constexpr void
fill(It first, It last, T const & value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

} // namespace um2
