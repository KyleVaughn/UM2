#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
addressof(T & arg) noexcept -> T *
{
  return __builtin_addressof(arg);
}

} // namespace um2
