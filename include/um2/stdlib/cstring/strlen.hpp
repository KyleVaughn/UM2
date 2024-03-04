#pragma once

#include <um2/config.hpp>

namespace um2
{

PURE HOSTDEV [[nodiscard]] inline constexpr auto
strlen(char const * str) noexcept -> uint64_t
{
  uint64_t i = 0;
  for (; str[i] != '\0'; ++i) { }
  return i;
} 

} // namespace um2
