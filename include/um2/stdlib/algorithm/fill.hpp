#pragma once

#include <um2/config.hpp>

namespace um2
{

//==============================================================================
// fill 
//==============================================================================

template <typename ForwardIt, typename T> 
HOSTDEV constexpr void
fill(ForwardIt first, ForwardIt last, T const & value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

} // namespace um2
