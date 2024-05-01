#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T, class U>
PURE HOSTDEV [[nodiscard]] constexpr auto
memcmp(T const * lhs, U const * rhs, uint64_t n) -> int
{
  auto count = n;
  while (count != 0) {
    if (*lhs < *rhs) {
      return -1;
    }
    if (*rhs < *lhs) {
      return 1;
    }
    --count;
    ++lhs;
    ++rhs;
  }
  return 0;
}

} // namespace um2
