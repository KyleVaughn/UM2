#pragma once

#include <um2/config.hpp>

namespace um2
{

// Not technically part of the standard library, but a similar function is found in
// libc++.

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std
is_pointer_in_range(T const * begin, T const * end, T const * ptr) noexcept -> bool
{
  return begin <= ptr && ptr < end;
}

} // namespace um2
