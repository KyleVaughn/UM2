#pragma once

#include <um2/stdlib/memory/addressof.hpp>

namespace um2
{

template <class T>
HOSTDEV inline constexpr auto
voidify(T & from) noexcept -> void *
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<void*>(static_cast<const volatile void*>(um2::addressof(from)));
}

} // namespace um2
