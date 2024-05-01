#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T>
CONST HOSTDEV [[nodiscard]] constexpr auto
abs(T x) noexcept -> T
{
  return x < 0 ? -x : x;
}

} // namespace um2
