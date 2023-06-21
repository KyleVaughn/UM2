#pragma once

#include <type_traits>

namespace um2
{

// -----------------------------------------------------------------------------
// forward
// -----------------------------------------------------------------------------
template <class T>
HOSTDEV [[nodiscard]] constexpr auto
forward(std::remove_reference_t<T> & t) noexcept -> T &&
{
  return static_cast<T &&>(t);
}

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
forward(std::remove_reference_t<T> && t) noexcept -> T &&
{
  static_assert(!std::is_lvalue_reference_v<T>,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<T &&>(t);
}

} // namespace um2
