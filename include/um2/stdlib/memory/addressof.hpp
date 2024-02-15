#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T>
HOSTDEV [[nodiscard]] inline constexpr auto
addressof(T & arg) noexcept -> T *
{
  return __builtin_addressof(arg);
}

} // namespace um2
