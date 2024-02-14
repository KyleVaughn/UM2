#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T, class Size>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::fill_n
fill_n(T * first, Size n, T const & value) noexcept -> T *
{
  for (; n > 0; ++first, --n) {
    *first = value;
  }
  return first;
}


template <std::random_access_iterator RandomIt, class T>
HOST constexpr void
fill(T * first, RandomIt last, T const & value) noexcept
{
  fill_n(first, last - first, value);
}

} // namespace um2
