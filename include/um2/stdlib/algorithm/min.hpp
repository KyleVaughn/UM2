#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
min(T const & a, T const & b) noexcept -> T const &
{
  return b < a ? b : a;
}

} // namespace um2
