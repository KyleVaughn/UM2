#pragma once

#include <um2/config.hpp>

#include <type_traits> // std::remove_reference_t

namespace um2
{

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
// NOLINTNEXTLINE(*-param-not-moved,*std-forward) // False positive.
move(T && t) noexcept -> std::remove_reference_t<T> &&
{
  return static_cast<std::remove_reference_t<T> &&>(t);
}

} // namespace um2
