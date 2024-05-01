#pragma once

#include <um2/config.hpp>

#include <type_traits> // std::remove_reference_t, std::is_lvalue_reference_v

namespace um2
{

template <class T>
HOSTDEV [[nodiscard]] inline constexpr auto
forward(std::remove_reference_t<T> & t) noexcept -> T &&
{
  // Forwards lvalues as either lvalues or as rvalues, depending on T.
  return static_cast<T &&>(t);
}

template <class T>
HOSTDEV [[nodiscard]] inline constexpr auto
forward(std::remove_reference_t<T> && t) noexcept -> T &&
{
  // Forwards rvalues as rvalues and prohibits forwarding of lvalues.
  static_assert(!std::is_lvalue_reference_v<T>,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<T &&>(t);
}

} // namespace um2
