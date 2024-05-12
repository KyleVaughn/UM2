#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/assert.hpp>

namespace um2
{

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T const &
{
  ASSERT_ASSUME(lo <= hi);
  // NOLINTNEXTLINE(*avoid-nested-conditional-operator) OK
  return v < lo ? lo : (hi < v ? hi : v);
}

} // namespace um2
