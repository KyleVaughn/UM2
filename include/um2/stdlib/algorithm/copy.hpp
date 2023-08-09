#pragma once

#include <um2/config.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// copy
// -----------------------------------------------------------------------------
// Copies the elements in the range, defined by [first, last), to another range
// beginning at d_first. The function begins by copying *first into *d_first
// and then increments both first and d_first. If first == last, the function
// does nothing.
//
// https://en.cppreference.com/w/cpp/algorithm/copy
//
// We use __restrict__ to tell the compiler that the ranges do not overlap.

template <typename InputIt, typename OutputIt>
HOSTDEV constexpr auto
copy(InputIt __restrict__ first, InputIt last, OutputIt __restrict__ d_first) noexcept
    -> OutputIt
{
  for (; first != last; ++first, ++d_first) {
    *d_first = *first;
  }
  return d_first;
}

} // namespace um2
