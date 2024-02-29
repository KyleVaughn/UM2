#pragma once

#include <um2/config.hpp>

namespace um2
{

template <class T>
PURE HOSTDEV [[nodiscard]] inline constexpr auto
min(T const & a, T const & b) noexcept -> T const &
{
  // min should not care about the scope of the arguments
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-stack-address"
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape,clang-diagnostic-return-stack-address)
  return b < a ? b : a;
#pragma GCC diagnostic pop
}

} // namespace um2
