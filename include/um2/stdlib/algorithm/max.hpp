#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T>
PURE HOSTDEV [[nodiscard]] inline constexpr auto
max(T const & a, T const & b) noexcept -> T const &
{
  return a < b ? b : a;
}

} // namespace um2
