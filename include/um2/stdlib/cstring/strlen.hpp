#pragma once

#include <um2/config.hpp>

namespace um2
{

PURE HOSTDEV [[nodiscard]] inline constexpr auto
strlen(char const * str) noexcept -> uint64_t
{
    char const * end = str;
    while (*end != '\0') {
      ++end;
    }
    return static_cast<uint64_t>(end - str); 
} 

} // namespace um2
