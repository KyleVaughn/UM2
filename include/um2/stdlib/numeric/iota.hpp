#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class ForwardIt, class T>
HOSTDEV constexpr void
iota(ForwardIt first, ForwardIt last, T value) noexcept
{
  for (; first != last; ++first, ++value) {
    *first = value;
  }
}

} // namespace um2
