#pragma once

#include <type_traits>

namespace um2
{

// -----------------------------------------------------------------------------
// move
// -----------------------------------------------------------------------------
// https://en.cppreference.com/w/cpp/utility/move

template <class T>
HOSTDEV constexpr auto
move(T && t) noexcept -> std::remove_reference_t<T> &&
{
  return static_cast<std::remove_reference_t<T> &&>(t);
}

} // namespace um2
